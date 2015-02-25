namespace Core
{
	enum TEXTENCODE : uint8
	{
		TEXTENCODE_UTF8 = 1,
		TEXTENCODE_UTF16LE = 2,
		TEXTENCODE_UTF16BE = 3,
		TEXTENCODE_UTF16 = 4, // Use native byte order
		TEXTENCODE_ANY = 5, // sqlite3_create_function only
		TEXTENCODE_UTF16_ALIGNED = 8, // sqlite3_create_collation only
	};
	__device__ inline void operator|=(TEXTENCODE &a, int b) { a = (TEXTENCODE)(a | b); }
	__device__ inline void operator&=(TEXTENCODE &a, int b) { a = (TEXTENCODE)(a & b); }
	__device__ inline TEXTENCODE operator|(TEXTENCODE a, TEXTENCODE b) { return (TEXTENCODE)((int)a | (int)b); }
	__device__ inline TEXTENCODE operator&(TEXTENCODE a, TEXTENCODE b) { return (TEXTENCODE)((int)a & (int)b); }

#define ConvertEx_GetVarint32(A,B) \
	(uint8)((*(A)<(uint8)0x80)?((B)=(uint32)*(A)),1:\
	ConvertEx::GetVarint32((A),(uint32 *)&(B)))
#define ConvertEx_PutVarint32(A,B) \
	(uint8)(((uint32)(B)<(uint32)0x80)?(*(A)=(unsigned char)(B)),1:\
	ConvertEx::PutVarint32((A),(B)))
	class ConvertEx
	{
	public:
#pragma region Varint
		__device__ static int PutVarint(unsigned char *p, uint64 v);
		__device__ static int PutVarint32(unsigned char *p, uint32 v);
		__device__ static uint8 GetVarint(const unsigned char *p, uint64 *v);
		__device__ static uint8 GetVarint32(const unsigned char *p, uint32 *v);
		__device__ static int GetVarintLength(uint64 v);
#pragma endregion
#pragma region AtoX
		__device__ static bool Atof(const char *z, double *out, int length, TEXTENCODE encode);
		__device__ static int Atoi64(const char *z, int64 *out, int length, TEXTENCODE encode);
		__device__ static bool Atoi(const char *z, int *out);
		__device__ static inline int Atoi(const char *z)
		{
			int out = 0;
			if (z) Atoi(z, &out);
			return out;
		}
#pragma endregion

		__device__ inline static uint16 Get2nz(const uint8 *p) { return ((( (int)((p[0]<<8) | p[1]) -1)&0xffff)+1); }
		__device__ inline static uint16 Get2(const uint8 *p) { return (p[0]<<8) | p[1]; }
		__device__ inline static void Put2(unsigned char *p, uint32 v)
		{
			p[0] = (uint8)(v>>8);
			p[1] = (uint8)v;
		}
		__device__ inline static uint32 Get4(const uint8 *p) { return (p[0]<<24) | (p[1]<<16) | (p[2]<<8) | p[3]; }
		__device__ inline static void Put4(unsigned char *p, uint32 v)
		{
			p[0] = (uint8)(v>>24);
			p[1] = (uint8)(v>>16);
			p[2] = (uint8)(v>>8);
			p[3] = (uint8)v;
		}

#pragma region From: Pragma_c
		__device__ static uint8 GetSafetyLevel(const char *z, int omitFull, uint8 dflt);
		__device__ static bool GetBoolean(const char *z, uint8 dflt);
#pragma region

	};
}
