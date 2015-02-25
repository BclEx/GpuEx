//bitvec.c
namespace Core
{
#define BITVEC_SZ 512
#define BITVEC_USIZE (((BITVEC_SZ - (3 * sizeof(uint32))) / sizeof(Bitvec *)) * sizeof(Bitvec *))
#define BITVEC_SZELEM 8
#define BITVEC_NELEM (BITVEC_USIZE / sizeof(uint8))
#define BITVEC_NBIT (BITVEC_NELEM * BITVEC_SZELEM)
#define BITVEC_NINT (BITVEC_USIZE / sizeof(uint))
#define BITVEC_MXHASH (BITVEC_NINT / 2)
#define BITVEC_HASH(X) (((X) * 1) % BITVEC_NINT)
#define BITVEC_NPTR (BITVEC_USIZE / sizeof(Bitvec *))

	struct Bitvec
	{
	private:
		uint32 _size;      // Maximum bit index.  Max iSize is 4,294,967,296.
		uint32 _set;       // Number of bits that are set - only valid for aHash element.  Max is BITVEC_NINT.  For BITVEC_SZ of 512, this would be 125.
		uint32 _divisor;   // Number of bits handled by each apSub[] entry.
		// Should >=0 for apSub element. */
		// Max iDivisor is max(uint32) / BITVEC_NPTR + 1.
		// For a BITVEC_SZ of 512, this would be 34,359,739.
		union
		{
			uint8 Bitmap[BITVEC_NELEM]; // Bitmap representation
			uint32 Hash[BITVEC_NINT];	// Hash table representation
			Bitvec *Sub[BITVEC_NPTR];	// Recursive representation
		} u;
	public:
		__device__ Bitvec(uint32 size);
		__device__ bool Get(uint32 index);
		__device__ RC Set(uint32 index);
		__device__ void Clear(uint32 index, void *buffer);
		__device__ static inline void Destroy(Bitvec *p)
		{
			if (!p)
				return;
			if (p->_divisor)
				for (unsigned int index = 0; index < BITVEC_NPTR; index++)
					Destroy(p->u.Sub[index]);
			_free(p);
		}
		__device__ inline uint32 get_Length() { return _size; }
	};
}
