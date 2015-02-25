//bitvec.c
#include "Core.cu.h"

namespace Core
{
	__device__ Bitvec::Bitvec(uint32 size)
	{
		_assert(sizeof(Bitvec) == BITVEC_SZ);
		_memset(this, 0, sizeof(Bitvec));
		_size = size;		
	}

	__device__ bool Bitvec::Get(uint32 index)
	{
		if (index > _size || index == 0)
			return false;
		index--;
		Bitvec *p = this;
		while (p->_divisor)
		{
			uint32 bin = index / p->_divisor;
			index %= p->_divisor;
			p = p->u.Sub[bin];
			if (!p) return false;
		}
		if (p->_size <= BITVEC_NBIT)
			return ((p->u.Bitmap[index / BITVEC_SZELEM] & (1 << (index & (BITVEC_SZELEM - 1)))) != 0);
		uint32 h = BITVEC_HASH(index++);
		while (p->u.Hash[h])
		{
			if (p->u.Hash[h] == index) return true;
			h = (h + 1) % BITVEC_NINT;
		}
		return false;
	}

	__device__ RC Bitvec::Set(uint32 index)
	{
		_assert(index > 0);
		_assert(index <= _size);
		index--;
		Bitvec *p = this;
		while ((p->_size > BITVEC_NBIT) && p->_divisor)
		{
			uint32 bin = index / p->_divisor;
			index %= p->_divisor;
			if (!p->u.Sub[bin])
				if (!(p->u.Sub[bin] = new Bitvec(p->_divisor))) return RC_NOMEM;
			p = p->u.Sub[bin];
		}
		if (p->_size <= BITVEC_NBIT)
		{
			p->u.Bitmap[index / BITVEC_SZELEM] |= (1 << (index & (BITVEC_SZELEM - 1)));
			return RC_OK;
		}
		uint32 h = BITVEC_HASH(index++);
		// if there wasn't a hash collision, and this doesn't completely fill the hash, then just add it without worring about sub-dividing and re-hashing.
		if (!p->u.Hash[h])

			if (p->_set < (BITVEC_NINT - 1))
				goto bitvec_set_end;
			else
				goto bitvec_set_rehash;
		// there was a collision, check to see if it's already in hash, if not, try to find a spot for it
		do
		{
			if (p->u.Hash[h] == index) return RC_OK;
			h++;
			if (h >= BITVEC_NINT) h = 0;
		} while (p->u.Hash[h]);
		// we didn't find it in the hash.  h points to the first available free spot. check to see if this is going to make our hash too "full".
bitvec_set_rehash:
		if (p->_set >= BITVEC_MXHASH)
		{
			uint32 *values;
			if (!(values = (uint32 *)_stackalloc(nullptr, sizeof(p->u.Hash), false))) return RC_NOMEM;
			_memcpy(values, p->u.Hash, sizeof(p->u.Hash));
			_memset(p->u.Sub, 0, sizeof(p->u.Sub));
			p->_divisor = ((p->_size + BITVEC_NPTR - 1) / BITVEC_NPTR);
			int rc = p->Set(index);
			for (unsigned int j = 0; j < BITVEC_NINT; j++)
				if (values[j]) rc |= p->Set(values[j]);
			_stackfree(nullptr, values);
			return (RC)rc;
		}
bitvec_set_end:
		p->_set++;
		p->u.Hash[h] = index;
		return RC_OK;
	}

	__device__ void Bitvec::Clear(uint32 index, void *buffer)
	{
		_assert(index > 0);
		index--;
		Bitvec *p = this;
		while (p->_divisor)
		{
			uint32 bin = index / p->_divisor;
			index %= p->_divisor;
			p = p->u.Sub[bin];
			if (!p) return;
		}
		if (p->_size <= BITVEC_NBIT)
			p->u.Bitmap[index / BITVEC_SZELEM] &= ~(1 << (index & (BITVEC_SZELEM - 1)));
		else
		{
			uint32 *values = (uint32 *)buffer;
			_memcpy(values, p->u.Hash, sizeof(p->u.Hash));
			_memset(p->u.Hash, 0, sizeof(p->u.Hash));
			p->_set = 0;
			for (unsigned int j = 0; j < BITVEC_NINT; j++)
				if (values[j] && values[j] != (index + 1))
				{
					uint32 h = BITVEC_HASH(values[j] - 1);
					p->_set++;
					while (p->u.Hash[h])
					{
						h++;
						if (h >= BITVEC_NINT) h = 0;
					}
					p->u.Hash[h] = values[j];
				}
		}
	}

#pragma	region Tests
#ifdef TEST

#define SETBIT(V,I) V[(I) >> 3] |= (1 << ((I) & 7))
#define CLEARBIT(V,I) V[(I) >> 3] &= ~(1 << ((I) & 7))
#define TESTBIT(V,I) ((V[(I) >> 3] & (1 << ((I) & 7))) != 0)

	__device__ int Bitvec_BuiltinTest(int size, int *ops)
	{
		int rc = -1;
		// Allocate the Bitvec to be tested and a linear array of bits to act as the reference
		Bitvec *bitvec = new Bitvec(size);
		unsigned char *v = (unsigned char *)_alloc2((size + 7) / 8 + 1, true);
		void *tmpSpace = _alloc(BITVEC_SZ);
		int pc = 0;
		int i, nx, op;
		if (!bitvec || !v || !tmpSpace)
			goto bitvec_end;

		// Run the program
		while ((op = ops[pc]))
		{
			switch (op)
			{
			case 1:
			case 2:
			case 5:
				{
					nx = 4;
					i = ops[pc + 2] - 1;
					ops[pc + 2] += ops[pc + 3];
					break;
				}
			case 3:
			case 4: 
			default:
				{
					nx = 2;
					SysEx::PutRandom(sizeof(i), &i);
					break;
				}
			}
			if ((--ops[pc + 1]) > 0) nx = 0;
			pc += nx;
			i = (i & 0x7fffffff) % size;
			if (op & 1)
			{
				SETBIT(v, i + 1);
				if (op != 5)
					if (bitvec->Set(i + 1)) goto bitvec_end;
			}
			else
			{
				CLEARBIT(v, i + 1);
				bitvec->Clear(i + 1, tmpSpace);
			}
		}

		// Test to make sure the linear array exactly matches the Bitvec object.  Start with the assumption that they do
		// match (rc==0).  Change rc to non-zero if a discrepancy is found.
		rc = bitvec->Get(size + 1)
			+ bitvec->Get(0)
			+ (bitvec->get_Length() - size);
		for (i = 1; i <= size; i++)
		{
			if (TESTBIT(v, i) != bitvec->Get(i))
			{
				rc = i;
				break;
			}
		}

		// Free allocated structure
bitvec_end:
		_free(tmpSpace);
		_free(v);
		Bitvec::Destroy(bitvec);
		return rc;
	}

#endif
#pragma endregion
}
