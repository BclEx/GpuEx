using System;
using System.Diagnostics;
namespace Core
{
    public class HTable
    {
        public int Count;             // Number of entries with this hash
        public HashElem Chain;        // Pointer to first entry with this hash
    }

    public class HashElem
    {
        public HashElem Next, Prev;   // Next and previous elements in the table
        public object Data;           // Data associated with this element
        public string Key; public int KeyLength;               // Key associated with this element
    }

    public class Hash
    {
        public uint TableSize = 31;   // Number of buckets in the hash table
        public uint Count;            // Number of entries in this table
        public HashElem First;        // The first element of the array
        public HTable[] Table;
        public Hash memcpy()
        {
            return (this == null ? null : (Hash)MemberwiseClone());
        }

        public Hash()
        {
            Init();
        }

        public void Init()
        {
            First = null;
            Count = 0;
            TableSize = 0;
            Table = null;
        }

        public void Clear()
        {
            HashElem elem = First; // For looping over all elements of the table
            First = null;
            C._free(ref Table); Table = null;
            TableSize = 0;
            while (elem != null)
            {
                HashElem nextElem = elem.Next;
                C._free(ref elem);
                elem = nextElem;
            }
            Count = 0;
        }

        static uint GetHashCode(string key, int keyLength)
        {
            Debug.Assert(keyLength >= 0);
            int h = 0;
            int _z = 0;
            while (keyLength > 0) { h = (h << 3) ^ h ^ (_z < key.Length ? (int)char.ToLowerInvariant(key[_z++]) : 0); keyLength--; }
            return (uint)h;
        }

        static void InsertElement(Hash hash, HTable entry, HashElem newElem)
        {
            HashElem headElem; // First element already in entry
            if (entry != null)
            {
                headElem = (entry.Count != 0 ? entry.Chain : null);
                entry.Count++;
                entry.Chain = newElem;
            }
            else
                headElem = null;
            if (headElem != null)
            {
                newElem.Next = headElem;
                newElem.Prev = headElem.Prev;
                if (headElem.Prev != null) headElem.Prev.Next = newElem;
                else hash.First = newElem;
                headElem.Prev = newElem;
            }
            else
            {
                newElem.Next = hash.First;
                if (hash.First != null) hash.First.Prev = newElem;
                newElem.Prev = null;
                hash.First = newElem;
            }
        }

        static bool Rehash(Hash hash, uint newSize)
        {
#if MALLOC_SOFT_LIMIT
            if (newSize * sizeof(HTable) > MALLOC_SOFT_LIMIT)
                newSize = MALLOC_SOFT_LIMIT / sizeof(HTable);
            if (newSize == hash.TableSize) return false;
#endif
            // The inability to allocates space for a larger hash table is a performance hit but it is not a fatal error.  So mark the
            // allocation as a benign. Use sqlite3Malloc()/memset(0) instead of sqlite3MallocZero() to make the allocation, as sqlite3MallocZero()
            // only zeroes the requested number of bytes whereas this module will use the actual amount of space allocated for the hash table (which
            // may be larger than the requested amount).
            C._benignalloc_begin();
            HTable[] newTable = new HTable[newSize]; // The new hash table
            for (int i = 0; i < newSize; i++)
                newTable[i] = new HTable();
            C._benignalloc_end();
            if (newTable == null)
                return false;
            C._free(ref hash.Table);
            hash.Table = newTable;
            //hash.TableSize = newSize = SysEx.AllocSize(newTable) / sizeof(HTable);
            hash.TableSize = newSize;
            HashElem elem, nextElem;
            for (elem = hash.First, hash.First = null; elem != null; elem = nextElem)
            {
                uint h = GetHashCode(elem.Key, elem.KeyLength) % newSize;
                nextElem = elem.Next;
                InsertElement(hash, newTable[h], elem);
            }
            return true;
        }

        static HashElem FindElementGivenHash(Hash hash, string key, int keyLength, uint h)
        {
            HashElem elem; // Used to loop thru the element list
            int count; // Number of elements left to test
            if (hash.Table != null && hash.Table[h] != null)
            {
                HTable entry = hash.Table[h];
                elem = entry.Chain;
                count = entry.Count;
            }
            else
            {
                elem = hash.First;
                count = (int)hash.Count;
            }
            while (count-- > 0 && C._ALWAYS(elem != null))
            {
                if (elem.KeyLength == keyLength && elem.Key.Equals(key, StringComparison.OrdinalIgnoreCase))
                    return elem;
                elem = elem.Next;
            }
            return null;
        }

        static void RemoveElementGivenHash(Hash hash, ref HashElem elem, uint h)
        {
            HTable entry;
            if (elem.Prev != null)
                elem.Prev.Next = elem.Next;
            else
                hash.First = elem.Next;
            if (elem.Next != null)
                elem.Next.Prev = elem.Prev;
            if (hash.Table != null && hash.Table[h] != null)
            {
                entry = hash.Table[h];
                if (entry.Chain == elem)
                    entry.Chain = elem.Next;
                entry.Count--;
                Debug.Assert(entry.Count >= 0);
            }
            C._free(ref elem);
            hash.Count--;
            if (hash.Count == 0)
            {
                Debug.Assert(hash.First == null);
                Debug.Assert(hash.Count == 0);
                hash.Clear();
            }
        }

        public T Find<T>(string key, int keyLength, T nullType) where T : class
        {
            Debug.Assert(key != null);
            Debug.Assert(keyLength >= 0);
            uint h = (Table != null ? GetHashCode(key, keyLength) % TableSize : 0);
            HashElem elem = FindElementGivenHash(this, key, keyLength, h);
            return (elem != null ? (T)elem.Data : nullType);
        }

        public T Insert<T>(string key, int keyLength, T data) where T : class
        {
            Debug.Assert(key != null);
            Debug.Assert(keyLength >= 0);
            uint h = (Table != null ? GetHashCode(key, keyLength) % TableSize : 0); // the hash of the key modulo hash table size
            HashElem elem = FindElementGivenHash(this, key, keyLength, h); // Used to loop thru the element list
            if (elem != null)
            {
                T oldData = (T)elem.Data;
                if (data == null)
                    RemoveElementGivenHash(this, ref elem, h);
                else
                {
                    elem.Data = data;
                    elem.Key = key;
                    Debug.Assert(keyLength == elem.KeyLength);
                }
                return oldData;
            }
            if (data == null)
                return null;
            HashElem newElem = new HashElem();
            if (newElem == null)
                return null;
            newElem.Key = key;
            newElem.KeyLength = keyLength;
            newElem.Data = data;
            Count++;
            if (Count >= 10 && Count > 2 * TableSize)
            {
                if (Rehash(this, Count * 2))
                {
                    Debug.Assert(TableSize > 0);
                    h = GetHashCode(key, keyLength) % TableSize;
                }
            }
            InsertElement(this, (Table != null ? Table[h] : null), newElem);
            return null;
        }

    }
}