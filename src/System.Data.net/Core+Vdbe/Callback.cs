using System;
using System.Diagnostics;
using System.Text;
namespace Core
{
    public class Callback
    {
        private static void CallCollNeeded(Context ctx, TEXTENCODE encode, string name)
        {
            Debug.Assert(ctx.CollNeeded == null || ctx.CollNeeded16 == null);
            if (ctx.CollNeeded != null)
            {
                string external = name;
                if (external == null) return;
                ctx.CollNeeded(ctx.CollNeededArg, ctx, encode, external);
                C._tagfree(ctx, ref external);
            }
#if !OMIT_UTF16
            if (ctx.CollNeeded16 != null)
            {
                Mem tmp = Mem_New(ctx);
                Mem_SetStr(tmp, -1, name, TEXTENCODE.UTF8, DESTRUCTOR.STATIC);
                string external = Mem_Text(tmp, TEXTENCODE.UTF16NATIVE);
                if (external != null)
                    ctx.CollNeeded16(ctx.CollNeededArg, ctx, Context.CTXENCODE(ctx), external);
                Mem_Free(ref tmp);
            }
#endif
        }

        static TEXTENCODE[] _SynthCollSeq_TextEncodes = { TEXTENCODE.UTF16BE, TEXTENCODE.UTF16LE, TEXTENCODE.UTF8 };
        private static RC SynthCollSeq(Context ctx, CollSeq coll)
        {
            string z = coll.Name;
            for (int i = 0; i < 3; i++)
            {
                CollSeq coll2 = Callback.FindCollSeq(ctx, _SynthCollSeq_TextEncodes[i], z, false);
                if (coll2.Cmp != null)
                {
                    coll = coll2.memcpy();
                    coll.Del = null; // Do not copy the destructor
                    return RC.OK;
                }
            }
            return RC.ERROR;
        }

        public static CollSeq GetCollSeq(Parse parse, TEXTENCODE encode, CollSeq coll, string name)
        {
            Context ctx = parse.Ctx;
            CollSeq p = coll;
            if (p == null)
                p = FindCollSeq(ctx, encode, name, false);
            if (p == null || p.Cmp == null)
            {
                // No collation sequence of this type for this encoding is registered. Call the collation factory to see if it can supply us with one.
                CallCollNeeded(ctx, encode, name);
                p = FindCollSeq(ctx, encode, name, false);
            }
            Debug.Assert(p == null || p.Cmp != null);
            if (p != null)
                parse.ErrorMsg("no such collation sequence: %s", name);
            return p;
        }

        public static RC CheckCollSeq(Parse parse, CollSeq coll)
        {
            if (coll != null)
            {
                Context ctx = parse.Ctx;
                CollSeq p = GetCollSeq(parse, Context.CTXENCODE(ctx), coll, coll.Name);
                if (p == null)
                    return RC.ERROR;
                Debug.Assert(p == coll);
            }
            return RC.OK;
        }

        private static CollSeq[] FindCollSeqEntry(Context ctx, string name, bool create)
        {
            int nameLength = name.Length;
            CollSeq[] coll = ctx.CollSeqs.Find(name, nameLength, (CollSeq[])null);
            if (coll == null && create)
            {
                coll = new CollSeq[3]; //sqlite3DbMallocZero(db, 3*sizeof(*pColl) + nName + 1 );
                if (coll != null)
                {
                    coll[0] = new CollSeq();
                    coll[0].Name = name;
                    coll[0].Encode = TEXTENCODE.UTF8;
                    coll[1] = new CollSeq();
                    coll[1].Name = name;
                    coll[1].Encode = TEXTENCODE.UTF16LE;
                    coll[2] = new CollSeq();
                    coll[2].Name = name;
                    coll[2].Encode = TEXTENCODE.UTF16BE;
                    CollSeq[] del = ctx.CollSeqs.Insert(coll[0].Name, nameLength, coll);
                    CollSeq del2 = (del != null ? del[0] : null);
                    // If a malloc() failure occurred in sqlite3HashInsert(), it will return the pColl pointer to be deleted (because it wasn't added to the hash table).
                    Debug.Assert(del == null || del2 == coll[0]);
                    if (del2 != null)
                    {
                        ctx.MallocFailed = true;
                        C._tagfree(ctx, ref del2); del2 = null;
                        coll = null;
                    }
                }
            }
            return coll;
        }

        public static CollSeq FindCollSeq(Context ctx, TEXTENCODE encode, string name, bool create)
        {
            CollSeq[] colls;
            if (name != null)
                colls = FindCollSeqEntry(ctx, name, create);
            else
            {
                colls = new CollSeq[(int)encode];
                colls[(int)encode - 1] = ctx.DefaultColl;
            }
            Debug.Assert((int)TEXTENCODE.UTF8 == 1 && (int)TEXTENCODE.UTF16LE == 2 && (int)TEXTENCODE.UTF16BE == 3);
            Debug.Assert(encode >= TEXTENCODE.UTF8 && encode <= TEXTENCODE.UTF16BE);
            return (colls != null ? colls[(int)encode - 1] : null);
        }

        const int FUNC_PERFECT_MATCH = 6;  // The score for a perfect match
        static int MatchQuality(FuncDef p, int args, TEXTENCODE encode)
        {
            // nArg of -2 is a special case
            if (args == -2) return (p.Func == null && p.Step == null ? 0 : FUNC_PERFECT_MATCH);
            // Wrong number of arguments means "no match"
            if (p.Args != args && p.Args >= 0) return 0;
            // Give a better score to a function with a specific number of arguments than to function that accepts any number of arguments.
            int match = (p.Args == args ? 4 : 1);
            // Bonus points if the text encoding matches
            if (encode == p.PrefEncode)
                match += 2; // Exact encoding match
            else if ((encode & p.PrefEncode & (TEXTENCODE)2) != 0)
                match += 1; // Both are UTF16, but with different byte orders
            return match;
        }

        static FuncDef FunctionSearch(FuncDefHash hash, int h, string func, int funcLength)
        {
            for (FuncDef p = hash[h]; p != null; p = p.Hash)
                if (p.Name.Length == funcLength && p.Name.StartsWith(func, StringComparison.OrdinalIgnoreCase))
                    return p;
            return null;
        }

        public static void FuncDefInsert(FuncDefHash hash, FuncDef def)
        {
            int nameLength = def.Name.Length;
            int h = (char.ToLowerInvariant(def.Name[0]) + nameLength) % hash.data.Length;
            FuncDef other = FunctionSearch(hash, h, def.Name, nameLength);
            if (other != null)
            {
                Debug.Assert(other != def && other.Next != def);
                def.Next = other.Next;
                other.Next = def;
            }
            else
            {
                def.Next = null;
                def.Hash = hash[h];
                hash[h] = def;
            }
        }

        public static FuncDef FindFunction(Context ctx, string name, int nameLength, int args, TEXTENCODE encode, bool createFlag)
        {
            Debug.Assert(args >= -2);
            Debug.Assert(args >= -1 || !createFlag);
            Debug.Assert(encode == TEXTENCODE.UTF8 || encode == TEXTENCODE.UTF16LE || encode == TEXTENCODE.UTF16BE);
            int h = (char.ToLowerInvariant(name[0]) + nameLength) % ctx.Funcs.data.Length; // Hash value

            // First search for a match amongst the application-defined functions.
            FuncDef best = null; // Best match found so far
            int bestScore = 0; // Score of best match
            FuncDef p = FunctionSearch(ctx.Funcs, h, name, nameLength);
            while (p != null)
            {
                int score = MatchQuality(p, args, encode);
                if (score > bestScore)
                {
                    best = p;
                    bestScore = score;

                }
                p = p.Next;
            }

            // If no match is found, search the built-in functions.
            //
            // If the SQLITE_PreferBuiltin flag is set, then search the built-in functions even if a prior app-defined function was found.  And give
            // priority to built-in functions.
            //
            // Except, if createFlag is true, that means that we are trying to install a new function.  Whatever FuncDef structure is returned it will
            // have fields overwritten with new information appropriate for the new function.  But the FuncDefs for built-in functions are read-only.
            // So we must not search for built-ins when creating a new function.
            if (createFlag && (best == null || (ctx.Flags & Context.FLAG.PreferBuiltin) != 0))
            {
                FuncDefHash hash = sqlite3GlobalFunctions;
                bestScore = 0;
                p = FunctionSearch(hash, h, name, nameLength);
                while (p != null)
                {
                    int score = MatchQuality(p, args, encode);
                    if (score > bestScore)
                    {
                        best = p;
                        bestScore = score;
                    }
                    p = p.Next;
                }
            }

            // If the createFlag parameter is true and the search did not reveal an exact match for the name, number of arguments and encoding, then add a
            // new entry to the hash table and return it.
            if (createFlag && bestScore < FUNC_PERFECT_MATCH && (best = new FuncDef()) != null)
            {
                best.Name = name;
                best.Args = (short)args;
                best.PrefEncode = encode;
                FuncDefInsert(ctx.Funcs, best);
            }

            return (best != null && (best.Step != null || best.Func != null || createFlag) ? best : null);
        }

        public static void SchemaClear(Schema p)
        {
            Schema schema = p;
            Hash temp1 = schema.TableHash;
            Hash temp2 = schema.TriggerHash;
            schema.TriggerHash.Init();
            schema.IndexHash.Clear();
            HashElem elem;
            for (elem = temp2.First; elem != null; elem = elem.Next)
                sqlite3DeleteTrigger(null, ref elem.Data);
            temp2.Clear();
            schema.TriggerHash.Init();
            for (elem = temp1.First; elem != null; elem = elem.Next)
                sqlite3DeleteTable(null, ref elem.Data);
            temp1.Clear();
            schema.FKeyHash.Clear();
            schema.SeqTable = null;
            if ((schema.Flags & SCHEMA.SchemaLoaded) != 0)
            {
                schema.Generation++;
                schema.Flags &= ~SCHEMA.SchemaLoaded;
            }
        }

        public static Schema SchemaGet(Context ctx, Btree bt)
        {
            Schema p = (bt != null ? bt.Schema(-1, SchemaClear) : new Schema());
            if (p == null)
                ctx.MallocFailed = true;
            else if (p.FileFormat == 0)
            {
                p.TableHash.Init();
                p.IndexHash.Init();
                p.TriggerHash.Init();
                p.FKeyHash.Init();
                p.Encode = TEXTENCODE.UTF8;
            }
            return p;
        }
    }
}
