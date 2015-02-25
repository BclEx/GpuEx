// analyze.c
#region OMIT_ANALYZE
#if !OMIT_ANALYZE
using System;
using System.Diagnostics;
using System.Text;
using tRowcnt = System.UInt32; // 32-bit is the default

namespace Core.Command
{
    public class Analyze
    {
        public struct Table_t
        {
            public string Name;
            public string Cols;
            public Table_t(string name, string cols)
            {
                Name = name;
                Cols = cols;
            }
        }
        static Table_t[] _tables = new Table_t[]
        {
            new Table_t("sqlite_stat1", "tbl,idx,stat"),
#if ENABLE_STAT3
            new Table_t( "sqlite_stat3", "tbl,idx,new,sampleno,sample" ),
#endif
        };

        static void OpenStatTable(Parse parse, int db, int statCur, string where_, string whereType)
        {
            int[] roots = new int[] { 0, 0 };
            byte[] createTbls = new byte[] { 0, 0 };

            Context ctx = parse.Ctx;
            Vdbe v = parse.GetVdbe();
            if (v == null) return;
            Debug.Assert(Btree.HoldsAllMutexes(ctx));
            Debug.Assert(v.Ctx == ctx);
            Context.DB dbObj = ctx.DBs[db];

            for (int i = 0; i < _tables.Length; i++)
            {
                string tableName = _tables[i].Name;
                Table stat;
                if ((stat = Parse.FindTable(ctx, tableName, dbObj.Name)) == null)
                {
                    // The sqlite_stat[12] table does not exist. Create it. Note that a side-effect of the CREATE TABLE statement is to leave the rootpage 
                    // of the new table in register pParse.regRoot. This is important because the OpenWrite opcode below will be needing it.
                    parse.NestedParse("CREATE TABLE %Q.%s(%s)", dbObj.Name, tableName, _tables[i].Cols);
                    roots[i] = parse.RegRoot;
                    createTbls[i] = Vdbe::OPFLAG_P2ISREG;
                }
                else
                {
                    // The table already exists. If zWhere is not NULL, delete all entries associated with the table zWhere. If zWhere is NULL, delete the
                    // entire contents of the table.
                    roots[i] = stat.Id;
                    sqlite3TableLock(parse, db, roots[i], 1, tableName);
                    if (where_ == null)
                        parse.NestedParse("DELETE FROM %Q.%s WHERE %s=%Q", dbObj.Name, tableName, whereType, where_);
                    else
                        v.AddOp2(OP.Clear, roots[i], db); // The sqlite_stat[12] table already exists.  Delete all rows.
                }
            }

            // Open the sqlite_stat[12] tables for writing.
            for (int i = 0; i < _tables.Length; i++)
            {
                v.AddOp3(OP.OpenWrite, statCur + i, roots[i], db);
                v.ChangeP4(-1, 3, Vdbe.P4T.INT32);
                v.ChangeP5(createTbls[i]);
            }
        }

        const int STAT3_SAMPLES = 24;

        class Stat3Accum
        {
            public tRowcnt Rows;		// Number of rows in the entire table
            public tRowcnt PSamples;	// How often to do a periodic sample
            public int Min;				// Index of entry with minimum nEq and hash
            public int MaxSamples;      // Maximum number of samples to accumulate
            public uint Prn;            // Pseudo-random number used for sampling
            public class Stat3Sample
            {
                public long Rowid;             // Rowid in main table of the key
                public tRowcnt Eq;             // sqlite_stat3.nEq
                public tRowcnt Lt;             // sqlite_stat3.nLt
                public tRowcnt DLt;            // sqlite_stat3.nDLt
                public bool IsPSample;         // True if a periodic sample
                public uint Hash;              // Tiebreaker hash
            };
            public array_t<Stat3Sample> a;  // An array of samples
        };

#if ENABLE_STAT3
        static void Stat3Init_(FuncContext fctx, int argc, Mem[][] argv)
        {
            tRowcnt rows = (tRowcnt)Vdbe.Value_Int64(argv[0]);
            int maxSamples = Vdbe.Value_Int(argv[1]);
            int n = maxSamples;
            Stat3Accum p = new Stat3Accum
            {
                a = new array_t<Stat3Accum.Stat3Sample>(new Stat3Accum.Stat3Sample[n]),
                Rows = rows,
                MaxSamples = maxSamples,
                PSamples = (uint)(rows / (maxSamples / 3 + 1) + 1),
            };
            if (p == null)
            {
                Vdbe.Result_ErrorNoMem(fctx);
                return;
            }
            SysEx.Randomness(-1, p.Prn);
            Vdbe.Result_Blob(fctx, p, -1, C._free);
        }
        static const FuncDef stat3InitFuncdef = new FuncDef
	    {
		    2,					// nArg
		    TEXTENCODE.UTF8,	// iPrefEnc
		    (FUNC)0,			// flags
		    null,               // pUserData
		    null,               // pNext
		    Stat3Init_,			// xFunc
		    null,               // xStep
		    null,               // xFinalize
		    "stat3_init",		// zName
		    null,               // pHash
		    null                // pDestructor
	    };

        static void Stat3Push_(FuncContext fctx, int argc, Mem[] argv)
	    {
		    tRowcnt eq = (tRowcnt)Vdbe.Value_Int64(argv[0]);
		    if (eq == 0) return;
		    tRowcnt lt = (tRowcnt)Vdbe.Value_Int64(argv[1]);
		    tRowcnt dLt = (tRowcnt)Vdbe.Value_Int64(argv[2]);
		    long rowid = Vdbe.Value_Int64(argv[3]);
		    Stat3Accum p = (Stat3Accum)Vdbe.Value_Blob(argv[4]);
		    bool isPSample = false;
		    bool doInsert = false;
		    int min = p.Min;
		    uint h = p.Prn = p.Prn*1103515245 + 12345;
		    if ((lt/p.PSamples) != ((eq+lt)/p.PSamples)) doInsert = isPSample = true;
		    else if (p.a.length < p.MaxSamples) doInsert = true;
		    else if (eq > p.a[min].Eq || (eq == p.a[min].Eq && h > p.a[min].Hash)) doInsert = true;
		    if (!doInsert) return;
		    Stat3Accum.Stat3Sample sample;
		    if (p.a.length == p.MaxSamples)
		    {
			    Debug.Assert(p.a.length - min - 1 >= 0);
			    C._memmove(p.a[min], p.a[min+1], sizeof(p.a[0])*(p.a.length-min-1));
			    sample = p.a[p.a.length-1];
		    }
		    else
			    sample = p.a[p.a.length++];
		    sample.Rowid = rowid;
		    sample.Eq = eq;
		    sample.Lt = lt;
		    sample.DLt = dLt;
		    sample.Hash = h;
		    sample.IsPSample = isPSample;

		    // Find the new minimum
		    if (p.a.length == p.MaxSamples)
		    {
                int sampleIdx = 0; sample = p.a[sampleIdx];
			    int i = 0;
			    while (sample.IsPSample)
			    {
				    i++;
				    sampleIdx++; sample = p.a[sampleIdx];
				    Debug.Assert(i < p.a.length);
			    }
			    eq = sample.Eq;
			    h = sample.Hash;
			    min = i;
			    for (i++, sampleIdx++; i < p.a.length; i++, sampleIdx++)
			    {
                    sample = p.a[sampleIdx];
				    if (sample.IsPSample) continue;
				    if (sample.Eq < eq || (sample.Eq == eq && sample.Hash < h))
				    {
					    min = i;
					    eq = sample.Eq;
					    h = sample.Hash;
				    }
			    }
			    p.Min = min;
		    }
	    }
        static const FuncDef Stat3PushFuncdef = new FuncDef
	    {
		    5,					// nArg
		    TEXTENCODE.UTF8,    // iPrefEnc
		    (FUNC)0,			// flags
		    null,               // pUserData
		    null,               // pNext
		    Stat3Push_,			// xFunc
		    null,               // xStep
		    null,               // xFinalize
		    "stat3_push",		// zName
		    null,               // pHash
		    null                // pDestructor
	    };

        static void Stat3Get_(FuncContext fctx, int argc, Mem[] argv)
        {
            int n = Vdbe.Value_Int(argv[1]);
            Stat3Accum p = (Stat3Accum)Vdbe.Value_Blob(argv[0]);
            Debug.Assert(p != null);
            if (p.a.length <= n) return;
            switch (argc)
            {
                case 2: Vdbe.Result_Int64(fctx, p.a[n].Rowid); break;
                case 3: Vdbe.Result_Int64(fctx, p.a[n].Eq); break;
                case 4: Vdbe.Result_Int64(fctx, p.a[n].Lt); break;
                default: Vdbe.Result_Int64(fctx, p.a[n].DLt); break;
            }
        }
        static const FuncDef _stat3GetFuncdef = new FuncDef
	    {
		    -1,					// nArg
		    TEXTENCODE.UTF8,	// iPrefEnc
		    (FUNC)0,			// flags
		    null,               // pUserData
		    null,               // pNext
		    Stat3Get_,			// xFunc
		    null,               // xStep
		    null,               // xFinalize
		    "stat3_get",		// zName
		    null,               // pHash
		    null                // pDestructor
	    };

#endif

        static void AnalyzeOneTable(Parse parse, Table table, Index onlyIdx, int statCurId, int memId)
        {
            Context ctx = parse.Ctx;      // Database handle
            int i;                       // Loop counter
            int regTabname = memId++;     // Register containing table name
            int regIdxname = memId++;     // Register containing index name
            int regSampleno = memId++;    // Register containing next sample number
            int regCol = memId++;         // Content of a column analyzed table
            int regRec = memId++;         // Register holding completed record
            int regTemp = memId++;        // Temporary use register
            int regRowid = memId++;       // Rowid for the inserted record

            Vdbe v = parse.GetVdbe(); // The virtual machine being built up
            if (v == null || C._NEVER(table == null))
                return;
            // Do not gather statistics on views or virtual tables or system tables
            if (table.Id == 0 || table.Name.StartsWith("sqlite_", StringComparison.OrdinalIgnoreCase))
                return;
            Debug.Assert(Btree.HoldsAllMutexes(ctx));
            int db = sqlite3SchemaToIndex(ctx, table.Schema); // Index of database containing pTab
            Debug.Assert(db >= 0);
            Debug.Assert(Btree.SchemaMutexHeld(ctx, db, null));
#if !OMIT_AUTHORIZATION
            if (Auth.Check(parse, AUTH.ANALYZE, table.Name, 0, ctx.DBs[db].Name))
                return;
#endif

            // Establish a read-lock on the table at the shared-cache level.
            sqlite3TableLock(parse, db, table.Id, 0, table.Name);

            int zeroRows = -1; // Jump from here if number of rows is zero
            int idxCurId = parse.Tabs++; // Cursor open on index being analyzed
            v.AddOp4(OP.String8, 0, regTabname, 0, table.Name, 0);
            for (Index idx = table.Index; idx != null; idx = idx.Next) // An index to being analyzed
            {
                if (onlyIdx != null && onlyIdx != idx) continue;
                v.NoopComment("Begin analysis of %s", idx.Name);
                int cols = idx.Columns.length;
                int[] chngAddrs = C._tagalloc<int>(ctx, cols); // Array of jump instruction addresses
                if (chngAddrs == null) continue;
                KeyInfo key = sqlite3IndexKeyinfo(parse, idx);
                if (memId + 1 + (cols * 2) > parse.Mems)
                    parse.Mems = memId + 1 + (cols * 2);

                // Open a cursor to the index to be analyzed.
                Debug.Assert(db == sqlite3SchemaToIndex(ctx, idx.Schema));
                v.AddOp4(OP.OpenRead, idxCurId, idx.Id, db, key, Vdbe.P4T.KEYINFO_HANDOFF);
                v.VdbeComment("%s", idx.Name);

                // Populate the registers containing the index names.
                v.AddOp4(OP.String8, 0, regIdxname, 0, idx.Name, 0);

#if ENABLE_STAT3
                bool once = false; // One-time initialization
                if (once)
                {
                    once = false;
                    sqlite3OpenTable(parse, tabCurId, db, table, OP.OpenRead);
                }
                v.AddOp2(OP.Count, idxCurId, regCount);
                v.AddOp2(OP.Integer, STAT3_SAMPLES, regTemp1);
                v.AddOp2(OP.Integer, 0, regNumEq);
                v.AddOp2(OP.Integer, 0, regNumLt);
                v.AddOp2(OP.Integer, -1, regNumDLt);
                v.AddOp3(OP.Null, 0, regSample, regAccum);
                v.AddOp4(OP.Function, 1, regCount, regAccum, (object)_stat3InitFuncdef, Vdbe.P4T.FUNCDEF);
                v.ChangeP5(2);
#endif

                // The block of memory cells initialized here is used as follows.
                //
                //    iMem:                
                //        The total number of rows in the table.
                //
                //    iMem+1 .. iMem+nCol: 
                //        Number of distinct entries in index considering the left-most N columns only, where N is between 1 and nCol, 
                //        inclusive.
                //
                //    iMem+nCol+1 .. Mem+2*nCol:  
                //        Previous value of indexed columns, from left to right.
                //
                // Cells iMem through iMem+nCol are initialized to 0. The others are initialized to contain an SQL NULL.
                for (i = 0; i <= cols; i++)
                    v.AddOp2(OP.Integer, 0, memId + i);
                for (i = 0; i < cols; i++)
                    v.AddOp2(OP.Null, 0, memId + cols + i + 1);

                // Start the analysis loop. This loop runs through all the entries in the index b-tree.
                int endOfLoop = v.MakeLabel(); // The end of the loop
                v.AddOp2(OP.Rewind, idxCurId, endOfLoop);
                int topOfLoop = v.CurrentAddr(); // The top of the loop
                v.AddOp2(OP.AddImm, memId, 1);

                int addrIfNot = 0; // address of OP_IfNot
                for (i = 0; i < cols; i++)
                {
                    v.AddOp3(OP.Column, idxCurId, i, regCol);
                    if (i == 0) // Always record the very first row
                        addrIfNot = v.AddOp1(OP.IfNot, memId + 1);
                    Debug.Assert(idx.CollNames != null && idx.CollNames[i] != null);
                    CollSeq coll = sqlite3LocateCollSeq(parse, idx.CollNames[i]);
                    chngAddrs[i] = v.AddOp4(OP.Ne, regCol, 0, memId + cols + i + 1, coll, Vdbe.P4T.COLLSEQ);
                    v.ChangeP5(SQLITE_NULLEQ);
                    v.VdbeComment("jump if column %d changed", i);
#if ENABLE_STAT3
                    if (i == 0)
                    {
                        v.AddOp2(OP.AddImm, regNumEq, 1);
                        v.VdbeComment("incr repeat count");
                    }
#endif
                }
                v.AddOp2(OP.Goto, 0, endOfLoop);
                for (i = 0; i < cols; i++)
                {
                    v.JumpHere(chngAddrs[i]); // Set jump dest for the OP_Ne
                    if (i == 0)
                    {
                        v.JumpHere(addrIfNot); // Jump dest for OP_IfNot
#if ENABLE_STAT3
                        v.AddOp4(OP.Function, 1, regNumEq, regTemp2, (object)Stat3PushFuncdef, Vdbe.P4T.FUNCDEF);
                        v.ChangeP5(5);
                        v.AddOp3(OP.Column, idxCurId, idx.Columns.length, regRowid);
                        v.AddOp3(OP.Add, regNumEq, regNumLt, regNumLt);
                        v.AddOp2(OP.AddImm, regNumDLt, 1);
                        v.AddOp2(OP.Integer, 1, regNumEq);
#endif
                    }
                    v.AddOp2(OP.AddImm, memId + i + 1, 1);
                    v.AddOp3(OP.Column, idxCurId, i, memId + cols + i + 1);
                }
                C._tagfree(ctx, chngAddrs);

                // Always jump here after updating the iMem+1...iMem+1+nCol counters
                v.ResolveLabel(endOfLoop);

                v.AddOp2(OP.Next, idxCurId, topOfLoop);
                v.AddOp1(OP.Close, idxCurId);
#if ENABLE_STAT3
                v.AddOp4(OP.Function, 1, regNumEq, regTemp2, (object)Stat3PushFuncdef, Vdbe.P4T.FUNCDEF);
                v.ChangeP5(5);
                v.AddOp2(OP.Integer, -1, regLoop);
                int shortJump = v.AddOp2(OP_AddImm, regLoop, 1); // Instruction address
                v.AddOp4(OP.Function, 1, regAccum, regTemp1, (object)Stat3GetFuncdef, Vdbe.P4T.FUNCDEF);
                v.ChangeP5(2);
                v.AddOp1(OP.IsNull, regTemp1);
                v.AddOp3(OP.NotExists, tabCurId, shortJump, regTemp1);
                v.AddOp3(OP.Column, tabCurId, idx->Columns[0], regSample);
                sqlite3ColumnDefault(v, table, idx->Columns[0], regSample);
                v.AddOp4(OP.Function, 1, regAccum, regNumEq, (object)Stat3GetFuncdef, Vdbe.P4T.FUNCDEF);
                v.ChangeP5(3);
                v.AddOp4(OP.Function, 1, regAccum, regNumLt, (object)Stat3GetFuncdef, Vdbe.P4T.FUNCDEF);
                v.ChangeP5(4);
                v.AddOp4(OP.Function, 1, regAccum, regNumDLt, (object)Stat3GetFuncdef, Vdbe.P4T.FUNCDEF);
                v.ChangeP5(5);
                v.AddOp4(OP.MakeRecord, regTabname, 6, regRec, "bbbbbb", 0);
                v.AddOp2(OP.NewRowid, statCurId + 1, regNewRowid);
                v.AddOp3(OP.Insert, statCurId + 1, regRec, regNewRowid);
                v.AddOp2(OP.Goto, 0, shortJump);
                v.JumpHere(shortJump + 2);
#endif

                // Store the results in sqlite_stat1.
                //
                // The result is a single row of the sqlite_stat1 table.  The first two columns are the names of the table and index.  The third column
                // is a string composed of a list of integer statistics about the index.  The first integer in the list is the total number of entries
                // in the index.  There is one additional integer in the list for each column of the table.  This additional integer is a guess of how many
                // rows of the table the index will select.  If D is the count of distinct values and K is the total number of rows, then the integer is computed
                // as:
                //        I = (K+D-1)/D
                // If K==0 then no entry is made into the sqlite_stat1 table.  
                // If K>0 then it is always the case the D>0 so division by zero is never possible.
                v.AddOp2(OP_SCopy, memId, regStat1);
                if (zeroRows < 0)
                    zeroRows = v.AddOp1(OP.IfNot, memId);
                for (i = 0; i < cols; i++)
                {
                    v.AddOp4(OP.String8, 0, regTemp, 0, " ", 0);
                    v.AddOp3(OP.Concat, regTemp, regStat1, regStat1);
                    v.AddOp3(OP.Add, memId, memId + i + 1, regTemp);
                    v.AddOp2(OP.AddImm, regTemp, -1);
                    v.AddOp3(OP.Divide, memId + i + 1, regTemp, regTemp);
                    v.AddOp1(OP.ToInt, regTemp);
                    v.AddOp3(OP.Concat, regTemp, regStat1, regStat1);
                }
                v.AddOp4(OP.MakeRecord, regTabname, 3, regRec, "aaa", 0);
                v.AddOp2(OP.NewRowid, statCurId, regNewRowid);
                v.AddOp3(OP.Insert, statCurId, regRec, regNewRowid);
                v.ChangeP5(OPFLAG_APPEND);
            }

            // If the table has no indices, create a single sqlite_stat1 entry containing NULL as the index name and the row count as the content.
            if (table.Index == null)
            {
                v.AddOp3(OP.OpenRead, idxCurId, table->Id, db);
                v.VdbeComment("%s", table->Name);
                v.AddOp2(OP.Count, idxCurId, regStat1);
                v.AddOp1(OP.Close, idxCurId);
                zeroRows = v.AddOp1(OP.IfNot, regStat1);
            }
            else
            {
                v.JumpHere(zeroRows);
                zeroRows = v.AddOp0(OP_Goto);
            }
            v.AddOp2(OP.Null, 0, regIdxname);
            v.AddOp4(OP.MakeRecord, regTabname, 3, regRec, "aaa", 0);
            v.AddOp2(OP.NewRowid, statCurId, regNewRowid);
            v.AddOp3(OP.Insert, statCurId, regRec, regNewRowid);
            v.ChangeP5(OPFLAG_APPEND);
            if (parse.Mems < regRec) parse.Mems = regRec;
            v.JumpHere(zeroRows);
        }

        static void LoadAnalysis(Parse parse, int db)
        {
            Vdbe v = parse.GetVdbe();
            if (v != null)
                v.AddOp1(OP.LoadAnalysis, db);
        }

        static void AnalyzeDatabase(Parse parse, int db)
        {
            Context ctx = parse.Ctx;
            Schema schema = ctx.DBs[db].Schema; // Schema of database iDb
            parse.BeginWriteOperation(0, db);
            int statCurId = parse.nTab;
            parse.Tabs += 2;
            openStatTable(parse, db, statCurId, null, null);
            int memId = parse.Mems + 1;
            Debug.Assert(Btree.SchemaMutexHeld(ctx, db, null));
            for (HashElem k = schema.TableHash.First; k != null; k = k.Next)
            {
                Table table = (Table)k.Data;
                AnalyzeOneTable(parse, table, null, statCurId, memId);
            }
            LoadAnalysis(parse, db);
        }

        static void AnalyzeTable(Parse parse, Table table, Index onlyIdx)
        {
            Debug.Assert(table != null && Btree.HoldsAllMutexes(parse.Ctx));
            int db = Btree.SchemaToIndex(parse.db, table.pSchema);
            parse.BeginWriteOperation(parse, 0, db);
            int statCurId = parse.Tabs;
            parse.Tabs += 2;
            if (onlyIdx != null)
                OpenStatTable(parse, db, statCurId, onlyIdx.Name, "idx");
            else
                OpenStatTable(parse, db, statCurId, table.Name, "tbl");
            AnalyzeOneTable(parse, table, onlyIdx, statCurId, parse.Mems + 1);
            LoadAnalysis(parse, db);
        }

        public static void Analyze(Parse parse, int null2, int null3) { Analyze(Parse, null, null); } // OVERLOADS, so I don't need to rewrite parse.c
        public static void Analyze(Parse parse, Token name1, Token name2)
        {
            Context ctx = parse.Ctx;

            // Read the database schema. If an error occurs, leave an error message and code in pParse and return NULL.
            Debug.Assert(Btree.HoldsAllMutexes(parse.Ctx));
            if (sqlite3ReadSchema(parse) != RC.OK)
                return;

            Debug.Assert(name2 != null || name1 == null);
            if (name1 == null)
            {
                // Form 1:  Analyze everything
                for (int i = 0; i < ctx.DBs.length; i++)
                {
                    if (i == 1) continue; // Do not analyze the TEMP database
                    AnalyzeDatabase(parse, i);
                }
            }
            else if (name2.length == 0)
            {
                // Form 2:  Analyze the database or table named
                int db = sqlite3FindDb(ctx, name1);
                if (db >= 0)
                    AnalyzeDatabase(parse, db);
                else
                {
                    string z = sqlite3NameFromToken(ctx, name1);
                    if (z != null)
                    {
                        Index idx;
                        Table table;
                        if ((idx = sqlite3FindIndex(ctx, z, null)) != null)
                            AnalyzeTable(parse, idx.Table, idx);
                        else if ((table = sqlite3LocateTable(parse, 0, z, null)) != null)
                            AnalyzeTable(parse, table, null);
                        C._tagfree(ctx, z);
                    }
                }
            }
            else
            {
                // Form 3: Analyze the fully qualified table name
                Token tableName = null;
                int db = sqlite3TwoPartName(parse, pName1, pName2, ref tableName);
                if (db >= 0)
                {
                    string dbName = ctx.DBs[db].Name;
                    string z = sqlite3NameFromToken(ctx, tableName);
                    if (z != null)
                    {
                        Index idx;
                        Table table;
                        if ((idx = sqlite3FindIndex(ctx, z, dbName)) != null)
                            AnalyzeTable(parse, idx.pTable, idx);
                        else if ((table = sqlite3LocateTable(parse, 0, z, dbName)) != null)
                            AnalyzeTable(parse, table, null);
                        C._tagfree(ctx, z);
                    }
                }
            }
        }

        public struct AnalysisInfo
        {
            public Context Ctx;
            public string Database;
        };

        static bool AnalysisLoader(object data, long argc, object Oargv, object notUsed)
        {
            string[] argv = (string[])Oargv;
            AnalysisInfo info = (AnalysisInfo)data;
            Debug.Assert(argc == 3);
            if (argv == null || argv[0] == null || argv[2] == null)
                return false;
            Table table = sqlite3FindTable(info.Ctx, argv[0], info.Database);
            if (table == null)
                return false;
            Index index = (!string.IsNullOrEmpty(argv[1]) ? sqlite3FindIndex(info.Ctx, argv[1], info.Database) : null);
            int n = (index != null ? index.Columns.length : 0);
            string z = argv[2];
            int zIdx = 0;
            for (int i = 0; z != null && i <= n; i++)
            {
                tRowcnt v = 0;
                int c;
                while (zIdx < z.Length && (c = z[zIdx]) >= '0' && c <= '9')
                {
                    v = v * 10 + c - '0';
                    zIdx++;
                }
                if (i == 0) table.RowEst = v;
                if (index == null) break;
                index.RowEsts[i] = v;
                if (zIdx < z.Length && z[zIdx] == ' ') zIdx++;
                if (string.Equals(z.Substring(zIdx), "unordered", StringComparison.OrdinalIgnoreCase))
                {
                    index.Unordered = true;
                    break;
                }
            }
            return false;
        }

        public static void DeleteIndexSamples(Context ctx, Index idx)
        {
#if ENABLE_STAT3
            if (idx.Samples != null)
            {
                for (int j = 0; j < idx.Samples.length; j++)
                {
                    IndexSample p = idx.Samples[j];
                    if (p.Type == TYPE.TEXT || p.Type == TYPE.BLOB)
                        C._tagfree(ctx, ref p.u.Z);
                }
                C._tagfree(ctx, ref idx.Samples);
            }
            if (ctx && ctx.BytesFreed == 0)
            {
                idx.Samples.length = 0;
                idx.Samples.data = null;
            }
#endif
        }

#if ENABLE_STAT3
        static RC LoadStat3(Context ctx, string dbName)
        {
            Debug.Assert(!ctx.Lookaside.Enabled);
            if (sqlite3FindTable(ctx, "sqlite_stat3", dbName) == null)
                return RC.OK;

            string sql = SysEx.Mprintf(ctx, "SELECT idx,count(*) FROM %Q.sqlite_stat3 GROUP BY idx", dbName); // Text of the SQL statement
            if (!sql)
                return RC_NOMEM;
            Vdbe stmt = null; // An SQL statement being run
            RC rc = Vdbe.Prepare(ctx, sql, -1, stmt, 0); // Result codes from subroutines
            C._tagfree(ctx, sql);
            if (rc) return rc;

            while (stmt.Step() == RC.ROW)
            {
                string indexName = (string)Vdbe.Column_Text(stmt, 0); // Index name
                if (!indexName) continue;
                int samplesLength = Vdbe.Column_Int(stmt, 1); // Number of samples
                Index idx = sqlite3FindIndex(ctx, indexName, dbName); // Pointer to the index object
                if (!idx) continue;
                _assert(idx->Samples.length == 0);
                idx.Samples.length = samplesLength;
                idx.Samples.data = new IndexSample[samplesLength];
                idx.AvgEq = idx.RowEsts[1];
                if (!idx->Samples.data)
                {
                    ctx->MallocFailed = true;
                    Vdbe.Finalize(stmt);
                    return RC.NOMEM;
                }
            }
            rc = Vdbe.Finalize(stmt);
            if (rc) return rc;

            sql = C._mtagprintf(ctx, "SELECT idx,neq,nlt,ndlt,sample FROM %Q.sqlite_stat3", dbName);
            if (!sql)
                return RC.NOMEM;
            rc = Vdbe.Prepare(ctx, sql, -1, &stmt, 0);
            C._tagfree(ctx, sql);
            if (rc) return rc;

            Index prevIdx = null; // Previous index in the loop
            int idxId = 0; // slot in pIdx->aSample[] for next sample
            while (stmt.Step() == RC.ROW)
            {
                string indexName = (string)Vdbe.Column_Text(stmt, 0); // Index name
                if (indexName == null) continue;
                Index idx = sqlite3FindIndex(ctx, indexName, dbName); // Pointer to the index object
                if (idx == null) continue;
                if (idx == prevIdx)
                    idxId++;
                else
                {
                    prevIdx = idx;
                    idxId = 0;
                }
                Debug.Assert(idxId < idx.Samples.length);
                IndexSample sample = idx.Samples[idxId]; // A slot in pIdx->aSample[]
                sample.Eq = (tRowcnt)Vdbe.Column_Int64(stmt, 1);
                sample.Lt = (tRowcnt)Vdbe.Column_Int64(stmt, 2);
                sample.DLt = (tRowcnt)Vdbe.Column_Int64(stmt, 3);
                if (idxId == idx.Samples.length - 1)
                {
                    tRowcnt sumEq;  // Sum of the nEq values
                    if (sample.DLt > 0)
                    {
                        for (int i = 0, sumEq = 0; i <= idxId - 1; i++) sumEq += idx.Samples[i].Eq;
                        idx.AvgEq = (sample.Lt - sumEq) / sample.DLt;
                    }
                    if (idx.AvgEq <= 0) idx.AvgEq = 1;
                }
                TYPE type = Vdbe.Column_Type(stmt, 4); // Datatype of a sample
                sample.Type = type;
                switch (type)
                {
                    case TYPE.INTEGER:
                        {
                            sample.u.I = Vdbe.Column_Int64(stmt, 4);
                            break;
                        }
                    case TYPE.FLOAT:
                        {
                            sample.u.R = Vdbe.Column_Double(stmt, 4);
                            break;
                        }
                    case TYPE.NULL:
                        {
                            break;
                        }
                    default: Debug.Assert(type == TYPE.TEXT || type == TYPE.BLOB);
                        {
                            string z = (string)(type == TYPE_BLOB ? Vdbe.Column_Blob(stmt, 4) : Vdbe.Column_Text(stmt, 4));
                            int n = (z ? Vdbe.Column_Bytes(stmt, 4) : 0);
                            sample.Bytes = n;
                            if (n < 1)
                                sample.u.Z = null;
                            else
                            {
                                sample.u.Z = C._tagalloc(ctx, n);
                                if (sample->u.Z == null)
                                {
                                    ctx.MallocFailed = true;
                                    Vdbe.Finalize(stmt);
                                    return RC.NOMEM;
                                }
                                Buffer.BlockCopy(sample.u.Z, z, n);
                            }
                        }
                }
            }
            return Vdbe.Finalize(stmt);
        }
#endif
        public static RC AnalysisLoad(Context ctx, int db)
        {
            Debug.Assert(db >= 0 && db < ctx.DBs.length);
            Debug.Assert(ctx.DBs[db].Bt != null);

            // Clear any prior statistics
            Debug.Assert(Btree.SchemaMutexHeld(ctx, db, null));
            for (HashElem i = ctx.DBs[db].Schema.IndexHash.First; i != null; i = i.Next)
            {
                Index idx = (Index)i.Data;
                sqlite3DefaultRowEst(idx);
                sqlite3DeleteIndexSamples(ctx, idx);
                idx.Samples.data = null;
            }

            // Check to make sure the sqlite_stat1 table exists
            AnalysisInfo sInfo = new AnalysisInfo();
            sInfo.Ctx = ctx;
            sInfo.Database = ctx.DBs[db].Name;
            if (sqlite3FindTable(ctx, "sqlite_stat1", sInfo.Database) == null)
                return RC.ERROR;

            // Load new statistics out of the sqlite_stat1 table
            string sql = C._mtagprintf(ctx, "SELECT tbl, idx, stat FROM %Q.sqlite_stat1", sInfo.Database);
            if (sql == null)
                rc = RC.NOMEM;
            else
            {
                rc = Vdbe.Exec(ctx, sql, AnalysisLoader, sInfo, 0);
                C._tagfree(ctx, ref sql);
            }

            // Load the statistics from the sqlite_stat3 table.
#if ENABLE_STAT3
            if (rc == RC_OK)
            {
                bool lookasideEnabled = ctx.Lookaside.Enabled;
                ctx.Lookaside.Enabled = false;
                rc = LoadStat3(ctx, sInfo.Database);
                ctx.Lookaside.Enabled = lookasideEnabled;
            }
#endif

            if (rc == RC.NOMEM)
                db.MallocFailed = true;
            return rc;
        }
    }
}
#endif
#endregion
