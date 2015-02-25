// random.c
#include "Core.cu.h"

namespace Core
{
	__device__ static _WSD struct Prng
	{
		unsigned char IsInit;
		unsigned char I;
		unsigned char J;
		unsigned char S[256];
	} g_prng;
#ifdef OMIT_WSD
	Prng *p = &_GLOBAL(Prng, g_prng);
#define _prng p[0]
#else
#define _prng g_prng
#endif

	__device__ static uint8 RandomByte()
	{
		// Initialize the state of the random number generator once, the first time this routine is called.  The seed value does
		// not need to contain a lot of randomness since we are not trying to do secure encryption or anything like that...
		//
		// Nothing in this file or anywhere else in SQLite does any kind of encryption.  The RC4 algorithm is being used as a PRNG (pseudo-random
		// number generator) not as an encryption device.
		unsigned char t;
		if (!_prng.IsInit)
		{
			char k[256];
			_prng.J = 0;
			_prng.I = 0;
			VSystem::FindVfs(nullptr)->Randomness(256, k);
			int i;
			for (i = 0; i < 256; i++)
				_prng.S[i] = (uint8)i;
			for (i = 0; i < 256; i++)
			{
				_prng.J += _prng.S[i] + k[i];
				t = _prng.S[_prng.J];
				_prng.S[_prng.J] = _prng.S[i];
				_prng.S[i] = t;
			}
			_prng.IsInit = true;
		}
		// Generate and return single random u8
		_prng.I++;
		t = _prng.S[_prng.I];
		_prng.J += t;
		_prng.S[_prng.I] = _prng.S[_prng.J];
		_prng.S[_prng.J] = t;
		t += _prng.S[_prng.I];
		return _prng.S[t];
	}

	__device__ void SysEx::PutRandom(int length, void *buffer) //: sqlite3_randomness
	{
		unsigned char *b = (unsigned char *)buffer;
#if THREADSAFE
		MutexEx mutex = MutexEx::Alloc(MutexEx::MUTEX_STATIC_PRNG);
		MutexEx::Enter(mutex);
#endif
		while (length--)
			*(b++) = RandomByte();
#if THREADSAFE
		MutexEx::Leave(mutex);
#endif
	}

#if !OMIT_BUILTIN_TEST
	__device__ static _WSD Prng *g_savedPrng = nullptr;
	__device__ void Random_PrngSaveState() { _memcpy<void>(&_GLOBAL(Prng, g_savedPrng), &_GLOBAL(Prng, g_prng), sizeof(g_prng)); }
	__device__ void Random_PrngRestoreState() { _memcpy<void>(&_GLOBAL(Prng, g_prng), &_GLOBAL(Prng, g_savedPrng), sizeof(g_prng)); }
	__device__ void Random_PrngResetState() { _prng.IsInit = false; }
#endif

}