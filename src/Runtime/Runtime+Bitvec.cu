#include "Runtime.h"

__device__ Bitvec *Bitvec::New(uint32 size)
{
	_assert(sizeof(Bitvec) == BITVEC_SZ);
	Bitvec *p = (Bitvec *)_allocZero(sizeof(Bitvec));
	if (p)
		p->_size = size;
	return p;
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

__device__ bool Bitvec::Set(uint32 index)
{
	_assert(index > 0);
	_assert(index <= _size);
	index--;
	Bitvec *p = this;
	while ((p->_size > BITVEC_NBIT) && p->_divisor)
	{
		uint32 bin = index / p->_divisor;
		index %= p->_divisor;
		if (!p->u.Sub[bin] && !(p->u.Sub[bin] = New(p->_divisor))) return false;
		p = p->u.Sub[bin];
	}
	if (p->_size <= BITVEC_NBIT)
	{
		p->u.Bitmap[index / BITVEC_SZELEM] |= (1 << (index & (BITVEC_SZELEM - 1)));
		return true;
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
		if (p->u.Hash[h] == index) return true;
		h++;
		if (h >= BITVEC_NINT) h = 0;
	} while (p->u.Hash[h]);
	// we didn't find it in the hash.  h points to the first available free spot. check to see if this is going to make our hash too "full".
bitvec_set_rehash:
	if (p->_set >= BITVEC_MXHASH)
	{
		uint32 *values = (uint32 *)_stackalloc(nullptr, sizeof(p->u.Hash));
		if (!values) return false;
		_memcpy(values, p->u.Hash, sizeof(p->u.Hash));
		_memset(p->u.Sub, 0, sizeof(p->u.Sub));
		p->_divisor = ((p->_size + BITVEC_NPTR - 1) / BITVEC_NPTR);
		bool rc = p->Set(index);
		for (unsigned int j = 0; j < BITVEC_NINT; j++)
			if (values[j]) rc |= p->Set(values[j]);
		_stackfree(nullptr, values);
		return rc;
	}
bitvec_set_end:
	p->_set++;
	p->u.Hash[h] = index;
	return true;
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