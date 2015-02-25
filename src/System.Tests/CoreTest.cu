#include <Core\Core.cu.h>

#pragma region Preamble

#if __CUDACC__
#define TEST(id) \
	__global__ void coreTest##id(void *r); \
	void coreTest##id##_host(cudaDeviceHeap &r) { coreTest##id<<<1, 1>>>(r.heap); cudaDeviceHeapSynchronize(r); } \
	__global__ void coreTest##id(void *r) \
{ \
	_runtimeSetHeap(r); \
	SysEx::Initialize();
#else
#define TEST(id) \
	__global__ void coreTest##id(void *r); \
	void coreTest##id##_host(cudaDeviceHeap &r) { coreTest##id(r.heap); cudaDeviceHeapSynchronize(r); } \
	__global__ void coreTest##id(void *r) \
{ \
	_runtimeSetHeap(r); \
	SysEx::Initialize();
#endif

#pragma endregion

namespace Core
{
	__device__ int Bitvec_BuiltinTest(int size, int *ops);
}

//////////////////////////////////////////////////

// bitvec
TEST(0) {
	// Test that sqlite3BitvecBuiltinTest correctly reports errors that are deliberately introduced.
	int op0[] = {5, 1, 1, 1, 0}; _assert(Bitvec_BuiltinTest(400, op0) == 1);
	int op1[] = {5, 1, 234, 1, 0}; _assert(Bitvec_BuiltinTest(400, op1) == 234);

	// Run test cases that set every bit in vectors of various sizes. for larger cases, this should cycle the bit vector representation from hashing into subbitmaps.
	// The subbitmaps should start as hashes then change to either subbitmaps or linear maps, depending on their size.
	int op2[] = {1, 400, 1, 1, 0}; _assert(Bitvec_BuiltinTest(400, op2) == 0);
	int op3[] = {1, 4000, 1, 1, 0}; _assert(Bitvec_BuiltinTest(4000, op3) == 0);
	int op4[] = {1, 40000, 1, 1, 0}; _assert(Bitvec_BuiltinTest(40000, op4) == 0);
	int op5[] = {1, 400000, 1, 1, 0}; _assert(Bitvec_BuiltinTest(400000, op5) == 0);

	// By specifying a larger increments, we spread the load around.
	int op6[] = {1, 400, 1, 7, 0}; _assert(Bitvec_BuiltinTest(400, op6) == 0);
	int op7[] = {1, 4000, 1, 7, 0}; _assert(Bitvec_BuiltinTest(4000, op7) == 0);
	int op8[] = {1, 40000, 1, 7, 0}; _assert(Bitvec_BuiltinTest(40000, op8) == 0);
	int op9[] = {1, 400000, 1, 7, 0}; _assert(Bitvec_BuiltinTest(400000, op9) == 0);

	// First fill up the bitmap with ones,  then go through and clear all the bits.  This will stress the clearing mechanism.
	int op10[] = {1, 400, 1, 1, 2, 400, 1, 1, 0}; _assert(Bitvec_BuiltinTest(400, op10) == 0);
	int op11[] = {1, 4000, 1, 1, 2, 4000, 1, 1, 0}; _assert(Bitvec_BuiltinTest(4000, op11) == 0);
	int op12[] = {1, 40000, 1, 1, 2, 40000, 1, 1, 0}; _assert(Bitvec_BuiltinTest(40000, op12) == 0);
	int op13[] = {1, 400000, 1, 1, 2, 400000, 1, 1, 0}; _assert(Bitvec_BuiltinTest(400000, op13) == 0);
	//
	int op14[] = {1, 400, 1, 1, 2, 400, 1, 7, 0}; _assert(Bitvec_BuiltinTest(400, op14) == 0);
	int op15[] = {1, 4000, 1, 1, 2, 4000, 1, 7, 0}; _assert(Bitvec_BuiltinTest(4000, op15) == 0);
	int op16[] = {1, 40000, 1, 1, 2, 40000, 1, 77, 0}; _assert(Bitvec_BuiltinTest(40000, op16) == 0);
	int op17[] = {1, 400000, 1, 1, 2, 400000, 1, 777, 0}; _assert(Bitvec_BuiltinTest(400000, op17) == 0);
	//
	int op18[] = {1, 5000, 100000, 1, 2, 400000, 1, 37, 0}; _assert(Bitvec_BuiltinTest(400000, op18) == 0);

	// Attempt to induce hash collisions.
	int op18a[] = {1, 60, -1, 124, 2, 5000, 1, 1, 0}; 
	int op18b[] = {1, 60, -1, 125, 2, 5000, 1, 1, 0}; 
	for (int op18i = 1; op18i <= 8; op18i++) {
		op18a[2] = op18i; _assert(Bitvec_BuiltinTest(5000, op18a) == 0);
		op18b[2] = op18i; _assert(Bitvec_BuiltinTest(5000, op18b) == 0);
	}

	// big and slow
	int op19[] = {1, 17000000, 1, 1, 2, 17000000, 1, 1, 0}; _assert(Bitvec_BuiltinTest(17000000, op19) == 0);

	// Test setting and clearing a random subset of bits.
	int op20[] = {3, 2000, 4, 2000, 0}; _assert(Bitvec_BuiltinTest(4000, op20) == 0);
	int op21[] = {
		3, 1000, 4, 1000, 3, 1000, 4, 1000, 3, 1000, 4, 1000, 
		3, 1000, 4, 1000, 3, 1000, 4, 1000, 3, 1000, 4, 1000, 0};
	_assert(Bitvec_BuiltinTest(4000, op21) == 0);
	int op22[] = {3, 10, 0}; _assert(Bitvec_BuiltinTest(400000, op22) == 0);
	int op23[] = {3, 10, 2, 4000, 1, 1, 0}; _assert(Bitvec_BuiltinTest(4000, op23) == 0);
	int op24[] = {3, 20, 2, 5000, 1, 1, 0}; _assert(Bitvec_BuiltinTest(5000, op24) == 0);
	int op25[] = {3, 60, 2, 50000, 1, 1, 0}; _assert(Bitvec_BuiltinTest(50000, op25) == 0);
	int op26[] = {
		1, 25, 121, 125,
		1, 50, 121, 125,
		2, 25, 121, 125,
		0};
	_assert(Bitvec_BuiltinTest(5000, op26) == 0);
}}

// bitvec failures
TEST(1) {
	/*
	// This procedure runs sqlite3BitvecBuiltinTest with argments "n" and "program".  But it also causes a malloc error to occur after the
	// "failcnt"-th malloc.  The result should be "0" if no malloc failure occurs or "-1" if there is a malloc failure.
	//proc bitvec_malloc_test {label failcnt n program} {
	//  do_test $label [subst {
	//    sqlite3_memdebug_fail $failcnt
	//    set x \[sqlite3BitvecBuiltinTest $n [list $program]\]
	//    set nFail \[sqlite3_memdebug_fail -1\]
	//    if {\$nFail==0} {
	//      set ::go 0
	//      set x -1
	//    }
	//    set x
	//  }] -1
	//}

	// Make sure malloc failures are handled sanily.
	//unset -nocomplain n
	//unset -nocomplain go
	//set go 1
	//save_prng_state
	//for {set n 0} {$go} {incr n} {
	//  restore_prng_state
	//  bitvec_malloc_test bitvec-3.1.$n $n 5000 {
	//      3 60 2 5000 1 1 3 60 2 5000 1 1 3 60 2 5000 1 1 0
	//  }
	//}
	//set go 1
	//for {set n 0} {$go} {incr n} {
	//  restore_prng_state
	//  bitvec_malloc_test bitvec-3.2.$n $n 5000 {
	//      3 600 2 5000 1 1 3 600 2 5000 1 1 3 600 2 5000 1 1 0
	//  }
	//}
	//set go 1
	//for {set n 1} {$go} {incr n} {
	//  bitvec_malloc_test bitvec-3.3.$n $n 50000 {1 50000 1 1 0}
	//}
	*/
}}