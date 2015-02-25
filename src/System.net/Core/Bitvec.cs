using System;
using System.Diagnostics;
namespace Core
{
    public class Bitvec
    {
        private static int BITVEC_SZ = 512;
        private static int BITVEC_USIZE = (((BITVEC_SZ - (3 * sizeof(uint))) / 4) * 4);
        private const int BITVEC_SZELEM = 8;
        private static int BITVEC_NELEM = (int)(BITVEC_USIZE / sizeof(byte));
        private static int BITVEC_NBIT = (BITVEC_NELEM * BITVEC_SZELEM);
        private static uint BITVEC_NINT = (uint)(BITVEC_USIZE / sizeof(uint));
        private static int BITVEC_MXHASH = (int)(BITVEC_NINT / 2);
        private static uint BITVEC_HASH(uint X) { return (uint)(((X) * 1) % BITVEC_NINT); }
        private static int BITVEC_NPTR = (int)(BITVEC_USIZE / 4);

        private uint _size;      // Maximum bit index.  Max iSize is 4,294,967,296.
        private uint _set;       // Number of bits that are set - only valid for aHash element.  Max is BITVEC_NINT.  For BITVEC_SZ of 512, this would be 125.
        private uint _divisor;   // Number of bits handled by each apSub[] entry.
        // Should >=0 for apSub element.
        // Max iDivisor is max(uint32) / BITVEC_NPTR + 1.
        // For a BITVEC_SZ of 512, this would be 34,359,739.
        private class _u
        {
            public byte[] Bitmap = new byte[BITVEC_NELEM]; // Bitmap representation
            public uint[] Hash = new uint[BITVEC_NINT];     // Hash table representation
            public Bitvec[] Sub = new Bitvec[BITVEC_NPTR];  // Recursive representation
        }
        private _u u = new _u();

        public Bitvec(uint size)
        {
            _size = size;
        }

        public static implicit operator bool(Bitvec b) { return (b != null); }

        /// <summary>
        /// Check to see if the i-th bit is set.  Return true or false.
        /// </summary>
        /// <remarks>
        /// If i is out of range, then return false.
        /// </remarks>
        /// <param name="index"></param>
        /// <returns></returns>
        public bool Get(uint index)
        {
            if (index > _size || index == 0)
                return false;
            index--;
            var p = this;
            while (p._divisor != 0)
            {
                uint bin = index / p._divisor;
                index %= p._divisor;
                p = p.u.Sub[bin];
                if (p == null) return false;
            }
            if (p._size <= BITVEC_NBIT)
                return ((p.u.Bitmap[index / BITVEC_SZELEM] & (1 << (int)(index & (BITVEC_SZELEM - 1)))) != 0);
            var h = BITVEC_HASH(index++);
            while (p.u.Hash[h] != 0)
            {
                if (p.u.Hash[h] == index) return true;
                h = (h + 1) % BITVEC_NINT;
            }
            return false;
        }

        /// <summary>
        /// Set the i-th bit.  Return 0 on success and an error code if anything goes wrong.
        /// </summary>
        /// <remarks>
        /// This routine might cause sub-bitmaps to be allocated.  Failing to get the memory needed to hold the sub-bitmap is the only
        /// that can go wrong with an insert, assuming p and i are valid.
        /// 
        /// The calling function must ensure that the value for "i" is within range of the Bitvec object.
        /// </remarks>
        /// <param name="index"></param>
        /// <returns></returns>
        public RC Set(uint index)
        {
            Debug.Assert(index > 0);
            Debug.Assert(index <= _size);
            index--;
            var p = this;
            while (p._size > BITVEC_NBIT && p._divisor != 0)
            {
                uint bin = index / p._divisor;
                index %= p._divisor;
                if (p.u.Sub[bin] == null)
                    p.u.Sub[bin] = new Bitvec(p._divisor);
                p = p.u.Sub[bin];
            }
            if (p._size <= BITVEC_NBIT)
            {
                p.u.Bitmap[index / BITVEC_SZELEM] |= (byte)(1 << (int)(index & (BITVEC_SZELEM - 1)));
                return RC.OK;
            }
            var h = BITVEC_HASH(index++);
            // if there wasn't a hash collision, and this doesn't completely fill the hash, then just add it without worring about sub-dividing and re-hashing.
            if (p.u.Hash[h] == 0)
                if (p._set < (BITVEC_NINT - 1))
                    goto bitvec_set_end;
                else
                    goto bitvec_set_rehash;
            // there was a collision, check to see if it's already in hash, if not, try to find a spot for it 
            do
            {
                if (p.u.Hash[h] == index) return RC.OK;
                h++;
                if (h >= BITVEC_NINT) h = 0;
            } while (p.u.Hash[h] != 0);
        // we didn't find it in the hash.  h points to the first available free spot. check to see if this is going to make our hash too "full".
        bitvec_set_rehash:
            if (p._set >= BITVEC_MXHASH)
            {
                var values = new uint[BITVEC_NINT];
                Buffer.BlockCopy(p.u.Hash, 0, values, 0, (int)BITVEC_NINT * sizeof(uint));
                Buffer.SetByte(p.u.Sub, 0, 0);
                p._divisor = (uint)((p._size + BITVEC_NPTR - 1) / BITVEC_NPTR);
                var rc = p.Set(index);
                for (uint j = 0; j < BITVEC_NINT; j++)
                    if (values[j] != 0) rc |= p.Set(values[j]);
                return rc;
            }
        bitvec_set_end:
            p._set++;
            p.u.Hash[h] = index;
            return RC.OK;
        }

        /// <summary>
        /// Clear the i-th bit
        /// </summary>
        /// <remarks>
        /// buffer must be a pointer to at least BITVEC_SZ bytes of temporary storage that Clear can use to rebuilt its hash table.
        /// </remarks>
        /// <param name="index"></param>
        /// <param name="buffer"></param>
        public void Clear(uint index, uint[] buffer)
        {
            Debug.Assert(index > 0);
            index--;
            var p = this;
            while (p._divisor != 0)
            {
                uint bin = index / p._divisor;
                index %= p._divisor;
                p = p.u.Sub[bin];
                if (p == null) return;
            }
            if (p._size <= BITVEC_NBIT)
                p.u.Bitmap[index / BITVEC_SZELEM] &= (byte)~((1 << (int)(index & (BITVEC_SZELEM - 1))));
            else
            {
                var values = buffer;
                Buffer.BlockCopy(p.u.Hash, 0, values, 0, values.Length * sizeof(uint));
                Buffer.SetByte(p.u.Hash, 0, 0);
                p._set = 0;
                for (uint j = 0; j < BITVEC_NINT; j++)
                    if (values[j] != 0 && values[j] != (index + 1))
                    {
                        var h = BITVEC_HASH(values[j] - 1);
                        p._set++;
                        while (p.u.Hash[h] != 0)
                        {
                            h++;
                            if (h >= BITVEC_NINT) h = 0;
                        }
                        p.u.Hash[h] = values[j];
                    }
            }
        }

        public static void Destroy(ref Bitvec p)
        {
            if (p == null)
                return;
            if (p._divisor != 0)
                for (uint index = 0; index < BITVEC_NPTR; index++)
                    Destroy(ref p.u.Sub[index]);
        }

        public uint Length
        {
            get { return _size; }
        }
    }
}