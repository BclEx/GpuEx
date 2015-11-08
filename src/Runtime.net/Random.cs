// random.c
using System;
using System.Diagnostics;
using System.Text;

namespace Core
{
    public partial class SysEx
    {
        public class PrngType
        {
            public bool IsInit;
            public int I;
            public int J;
            public byte[] S = new byte[256];

            public PrngType _memcpy()
            {
                PrngType cp = (PrngType)MemberwiseClone();
                cp.S = new byte[S.Length];
                Array.Copy(S, cp.S, S.Length);
                return cp;
            }
        }
        public static PrngType _prng = new PrngType();

        public static byte RandomByte()
        {
            // Initialize the state of the random number generator once, the first time this routine is called.  The seed value does
            // not need to contain a lot of randomness since we are not trying to do secure encryption or anything like that...
            //
            // Nothing in this file or anywhere else in SQLite does any kind of encryption.  The RC4 algorithm is being used as a PRNG (pseudo-random
            // number generator) not as an encryption device.
            byte t;
            if (!_prng.IsInit)
            {
                byte[] k = new byte[256];
                _prng.J = 0;
                _prng.I = 0;
                VSystem.FindVfs(string.Empty).Randomness(256, k);
                int i;
                for (i = 0; i < 255; i++)
                    _prng.S[i] = (byte)i;
                for (i = 0; i < 255; i++)
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

        public static void PutRandom(int length, byte[] buffer, int offset)
        {
            long bufferIdx = System.DateTime.Now.Ticks;
#if THREADSAFE
            MutexEx mutex = MutexEx.Alloc(MutexEx.MUTEX.STATIC_PRNG);
            MutexEx.Enter(mutex);
#endif
            while (length-- > 0)
            {
                bufferIdx = (uint)((bufferIdx << 8) + RandomByte());
                buffer[offset++] = (byte)bufferIdx;
            }
#if THREADSAFE
            MutexEx.Leave(mutex);
#endif
        }

        public static void PutRandom(int length, ref long bufferIdx)
        {
            bufferIdx = 0;
            byte[] b = new byte[length];
#if THREADSAFE
            MutexEx mutex = MutexEx.Alloc(MutexEx.MUTEX.STATIC_PRNG);
            MutexEx.Enter(mutex);
#endif
            while (length-- > 0)
                bufferIdx = (uint)((bufferIdx << 8) + RandomByte());
#if THREADSAFE
            MutexEx.Leave(mutex);
#endif
        }

#if !OMIT_BUILTIN_TEST
        static PrngType _savedPrng = null;
        static void PrngSaveState() { _savedPrng = _prng._memcpy(); }
        static void PrngRestoreState() { _prng = _savedPrng._memcpy(); }
        static void PrngResetState() { _prng.IsInit = false; }
#endif

        internal static RC AutoInitialize()
        {
            throw new NotImplementedException();
        }
    }
}
