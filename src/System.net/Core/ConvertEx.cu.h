namespace Core
{
	class ConvertEx
	{
	public:
		//__device__ inline static uint16 Get2nz(const uint8 *p) { return ((( (int)((p[0]<<8) | p[1]) -1)&0xffff)+1); }
		//__device__ inline static uint16 Get2(const uint8 *p) { return (p[0]<<8) | p[1]; }
		//__device__ inline static void Put2(unsigned char *p, uint32 v)
		//{
		//	p[0] = (uint8)(v>>8);
		//	p[1] = (uint8)v;
		//}
		//__device__ inline static uint32 Get4(const uint8 *p) { return (p[0]<<24) | (p[1]<<16) | (p[2]<<8) | p[3]; }
		//__device__ inline static void Put4(unsigned char *p, uint32 v)
		//{
		//	p[0] = (uint8)(v>>24);
		//	p[1] = (uint8)(v>>16);
		//	p[2] = (uint8)(v>>8);
		//	p[3] = (uint8)v;
		//}

#pragma region From: Pragma_c
		__device__ static uint8 GetSafetyLevel(const char *z, int omitFull, uint8 dflt);
		__device__ static bool GetBoolean(const char *z, uint8 dflt);
#pragma region

	};
}
