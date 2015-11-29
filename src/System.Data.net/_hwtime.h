// This file contains inline asm code for retrieving "high-performance" counters for x86 class CPUs.
#ifndef _HWTIME_H_
#define _HWTIME_H_

#if __CUDA_ARCH__

__device__ __inline__ uint64 _hwtime() { return ((uint64)0); }

#else

// The following routine only works on pentium-class (or newer) processors. It uses the RDTSC opcode to read the cycle count value out of the
// processor and returns that value.  This can be used for high-res profiling.
#if (defined(__GNUC__) || defined(_MSC_VER)) && (defined(i386) || defined(__i386__) || defined(_M_IX86))

#if defined(__GNUC__)

__inline__ uint64 _hwtime()
{
	unsigned int lo, hi;
	__asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
	return (uint64)hi << 32 | lo;
}

#elif defined(_MSC_VER)

__declspec(naked) __inline uint64 __cdecl _hwtime()
{
	__asm {
		rdtsc
			ret; return value at EDX:EAX
	}
}

#endif

#elif (defined(__GNUC__) && defined(__x86_64__))

__inline__ uint64 _hwtime()
{
	unsigned long val;
	__asm__ __volatile__ ("rdtsc" : "=A" (val));
	return val;
}

#elif (defined(__GNUC__) && defined(__ppc__))

__inline__ uint64 _hwtime()
{
	unsigned long long retval;
	unsigned long junk;
	__asm__ __volatile__ ("\n\
						  1:      mftbu   %1\n\
						  mftb    %L0\n\
						  mftbu   %0\n\
						  cmpw    %0,%1\n\
						  bne     1b"
						  : "=r" (retval), "=r" (junk));
	return retval;
}

#else

#error Need implementation of sqlite3Hwtime() for your platform.

// To compile without implementing sqlite3Hwtime() for your platform, you can remove the above #error and use the following
// stub function.  You will lose timing support for many of the debugging and testing utilities, but it should at
// least compile and run.

#endif

#endif

#endif