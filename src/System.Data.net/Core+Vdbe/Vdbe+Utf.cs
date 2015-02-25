#define TRANSLATE_TRACE //1
using System;
using System.Diagnostics;
using System.Text;

namespace Core
{
    public partial class Vdbe
    {
#if !OMIT_UTF16
        public static RC MemTranslate(Mem mem, TEXTENCODE desiredEncode)
        {
            Debugger.Break(); // TODO -
            int len; // Maximum length of output string in bytes
            //unsigned char *zOut; // Output buffer
            //unsigned char *zIn; // Input iterator
            //unsigned char *zTerm; // End of input
            //unsigned char *z; // Output iterator
            //unsigned int c;

            Debug.Assert(mem.Ctx == null || MutexEx.Held(mem.Ctx.Mutex));
            Debug.Assert((mem.Flags & MEM.Str) != 0);
            Debug.Assert(mem.Encode != desiredEncode);
            Debug.Assert(mem.Encode != 0);
            Debug.Assert(mem.N >= 0);

#if TRANSLATE_TRACE && DEBUG
            //{
            //    char[] buf = new char[100];
            //    MemPrettyPrint(mem, zBuf);
            //    fprintf(stderr, "INPUT:  %s\n", zBuf);
            //}
#endif

            Debugger.Break();
            // If the translation is between UTF-16 little and big endian, then all that is required is to swap the byte order. This case is handled differently from the others.
            //if (mem.Encode != TEXTENCODE.UTF8 && desiredEncode != TEXTENCODE.UTF8)
            //{
            //    u8 temp;
            //    RC rc = MemMakeWriteable(mem);
            //    if (rc != RC.OK)
            //    {
            //        Debug.Assert(rc == RC.NOMEM);
            //        return RC.NOMEM;
            //    }
            //    in_ = mem.Z;
            //    term_ = in_[mem.N & ~1];
            //    while (in_ < term_)
            //    {
            //        temp = in_;
            //        in_ = (in_ + 1);
            //        in_++;
            //        in_++ = temp;
            //    }
            //    mem.Encode = desiredEncode;
            //    goto translate_out;
            //}

            // Set len to the maximum number of bytes required in the output buffer.
            if (desiredEncode == TEXTENCODE.UTF8)
            {
                // When converting from UTF-16, the maximum growth results from translating a 2-byte character to a 4-byte UTF-8 character.
                // A single byte is required for the output string nul-terminator.
                mem.N &= ~1;
                len = mem.N * 2 + 1;
            }
            else
            {
                // When converting from UTF-8 to UTF-16 the maximum growth is caused when a 1-byte UTF-8 character is translated into a 2-byte UTF-16
                // character. Two bytes are required in the output buffer for the nul-terminator.
                len = mem.N * 2 + 2;
            }

            Debugger.Break();
            // Set zIn to point at the start of the input buffer and zTerm to point 1 byte past the end.
            // Variable zOut is set to point at the output buffer, space obtained from sqlite3_malloc().

            //zIn = (u8*)pMem.z;
            //zTerm = &zIn[pMem->n];
            //zOut = sqlite3DbMallocRaw(pMem->db, len);
            //if( !zOut ){
            //  return SQLITE_NOMEM;
            //}
            //z = zOut;

            //if( pMem->enc==SQLITE_UTF8 ){
            //  if( desiredEnc==SQLITE_UTF16LE ){
            //    /* UTF-8 -> UTF-16 Little-endian */
            //    while( zIn<zTerm ){
            ///* c = sqlite3Utf8Read(zIn, zTerm, (const u8**)&zIn); */
            //READ_UTF8(zIn, zTerm, c);
            //      WRITE_UTF16LE(z, c);
            //    }
            //  }else{
            //    Debug.Assert( desiredEnc==SQLITE_UTF16BE );
            //    /* UTF-8 -> UTF-16 Big-endian */
            //    while( zIn<zTerm ){
            ///* c = sqlite3Utf8Read(zIn, zTerm, (const u8**)&zIn); */
            //READ_UTF8(zIn, zTerm, c);
            //      WRITE_UTF16BE(z, c);
            //    }
            //  }
            //  pMem->n = (int)(z - zOut);
            //  *z++ = 0;
            //}else{
            //  Debug.Assert( desiredEnc==SQLITE_UTF8 );
            //  if( pMem->enc==SQLITE_UTF16LE ){
            //    /* UTF-16 Little-endian -> UTF-8 */
            //    while( zIn<zTerm ){
            //      READ_UTF16LE(zIn, zIn<zTerm, c); 
            //      WRITE_UTF8(z, c);
            //    }
            //  }else{
            //    /* UTF-16 Big-endian -> UTF-8 */
            //    while( zIn<zTerm ){
            //      READ_UTF16BE(zIn, zIn<zTerm, c); 
            //      WRITE_UTF8(z, c);
            //    }
            //  }
            //  pMem->n = (int)(z - zOut);
            //}
            //*z = 0;
            //Debug.Assert( (pMem->n+(desiredEnc==SQLITE_UTF8?1:2))<=len );

            //sqlite3VdbeMemRelease(pMem);
            //pMem->flags &= ~(MEM_Static|MEM_Dyn|MEM_Ephem);
            //pMem->enc = desiredEnc;
            //pMem->flags |= (MEM_Term|MEM_Dyn);
            //pMem.z = (char*)zOut;
            //pMem.zMalloc = pMem.z;

            //translate_out:
#if TRANSLATE_TRACE && DEBUG
            //{
            //    char[] zBuf = new char[100];
            //    MemPrettyPrint(mem, zBuf);
            //    fprintf(stderr, "OUTPUT: %s\n", zBuf);
            //}
#endif
            return RC.OK;
        }

        public static RC MemHandleBom(Mem mem)
        {
            RC rc = RC.OK;
            TEXTENCODE bom = 0;
            Debug.Assert(mem.N >= 0);
            if (mem.N > 1)
            {
                byte[] b01 = new byte[2];
                Encoding.Unicode.GetBytes(mem.Z, 0, 1, b01, 0);
                if (b01[0] == 0xFE && b01[1] == 0xFF)
                    bom = TEXTENCODE.UTF16BE;
                if (b01[0] == 0xFF && b01[1] == 0xFE)
                    bom = TEXTENCODE.UTF16LE;
            }
            if (bom != 0)
            {
                rc = MemMakeWriteable(mem);
                if (rc == RC.OK)
                {
                    mem.N -= 2;
                    Debugger.Break(); // TODO -
                    //memmove(pMem.z, pMem.z[2], pMem.n);
                    //pMem.z[pMem.n] = '\0';
                    //pMem.z[pMem.n+1] = '\0';
                    mem.Flags |= MEM.Term;
                    mem.Encode = bom;
                }
            }
            return rc;
        }
#endif

#if !OMIT_UTF16
        public static string Utf16to8(Context ctx, string z, int bytes, TEXTENCODE encode)
        {
            Debugger.Break(); // TODO -
            Mem m = null; // Pool.Allocate_Mem();
            //  memset(&m, 0, sizeof(m));
            //  m.db = db;
            //  sqlite3VdbeMemSetStr(&m, z, nByte, enc, SQLITE_STATIC);
            //  sqlite3VdbeChangeEncoding(&m, SQLITE_UTF8);
            //  if( db.mallocFailed !=0{
            //    sqlite3VdbeMemRelease(&m);
            //    m.z = 0;
            //  }
            //  Debug.Assert( (m.flags & MEM_Term)!=0 || db.mallocFailed !=0);
            //  Debug.Assert( (m.flags & MEM_Str)!=0 || db.mallocFailed !=0);
            Debug.Assert((m.Flags & MEM.Dyn) != 0 || ctx.MallocFailed);
            Debug.Assert(m.Z != null || ctx.MallocFailed);
            return m.Z;
        }

#if ENABLE_STAT3
        public string Utf8to16(Context ctx, TEXTENCODE encode, string z, int n, ref int out_)
        {
            Mem m = new Mem();
            m.Ctx = ctx;
            MemSetStr(m, z, n, TEXTENCODE.UTF8, C.DESTRUCTOR_STATIC);
            if (MemTranslate(m, encode) != RC.OK)
            {
                Debug.Assert(ctx.MallocFailed);
                return null;
            }
            //Debug.Assert(m.Z == m.zMalloc);
            out_ = m.N;
            return m.Z;
        }
#endif
#endif
    }
}
