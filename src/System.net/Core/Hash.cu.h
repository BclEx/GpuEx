//hash.c
namespace Core
{
	//#define HASH_FIRST(H) ((H)->first)
	//#define HASH_NEXT(E) ((E)->next)
	//#define HASH_DATA(E) ((E)->data)
	//#define HASH_Key(E)    ((E)->pKey) // NOT USED */
	//#define HASH_Keysize(E) ((E)->nKey)  // NOT USED */
	//#define HASH_Count(H)  ((H)->count) // NOT USED */

	struct HashElem
	{
		HashElem *Next, *Prev;       // Next and previous elements in the table
		void *Data;                  // Data associated with this element
		const char *Key; int KeyLength;  // Key associated with this element
	};

	struct Hash
	{
		unsigned int TableSize;     // Number of buckets in the hash table
		unsigned int Count;			// Number of entries in this table
		HashElem *First;			// The first element of the array
		struct HTable
		{              
			int Count;              // Number of entries with this hash
			HashElem *Chain;        // Pointer to first entry with this hash
		} *Table; // the hash table

		__device__ Hash();
		__device__ void Init();
		__device__ void *Insert(const char *key, int keyLength, void *data);
		__device__ void *Find(const char *key, int keyLength);
		__device__ void Clear();
	};
}
