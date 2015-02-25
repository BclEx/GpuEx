using System;
using System.Diagnostics;
using System.Text;
using tRowcnt = System.UInt32; // 32-bit is the default

namespace Core.Command
{
    public partial class Insert
    {
        static void OpenTable(Parse p, int cur, int db, Table table, OP opcode)
        {
            Debug.Assert(!E.IsVirtual(table));
            Vdbe v = p.GetVdbe();
            Debug.Assert(opcode == OP.OpenWrite || opcode == OP.OpenRead);
            p.TableLock(db, table.Id, (opcode == OP.OpenWrite), table.Name);
            v.AddOp3(opcode, cur, table.Id, db);
            v.ChangeP4(-1, table.Cols.length, Vdbe.P4T.INT32);
            v.Comment("%s", table.Name);
        }

        public static string IndexAffinityStr(Vdbe v, Index index)
        {
            if (index.ColAff == null || index.ColAff[0] == '\0')
            {
                // The first time a column affinity string for a particular index is required, it is allocated and populated here. It is then stored as
                // a member of the Index structure for subsequent use.
                //
                // The column affinity string will eventually be deleted by sqliteDeleteIndex() when the Index structure itself is cleaned up.
                Table table = index.Table;
                Context ctx = v.Ctx;
                StringBuilder b = new StringBuilder(index.Columns.length + 2); //: _tagalloc(nullptr, index->Columns.length+2);
                if (b == null)
                {
                    ctx.MallocFailed = true;
                    return null;
                }
                for (int n = 0; n < index.Columns.length; n++)
                    b.Append(table.Cols[index.Columns[n]].Affinity);
                b.Append(AFF.NONE);
                b.Append('\0');
                index.ColAff = b.ToString();
            }
            return index.ColAff;
        }

        public static void TableAffinityStr(Vdbe v, Table table)
        {
		    // The first time a column affinity string for a particular table is required, it is allocated and populated here. It is then 
		    // stored as a member of the Table structure for subsequent use.
		    //
		    // The column affinity string will eventually be deleted by sqlite3DeleteTable() when the Table structure itself is cleaned up.
            if (table.ColAff == null)
            {
                Context ctx = v.Ctx;
                StringBuilder b = new StringBuilder(table.Cols.length + 1);// (char)sqlite3DbMallocRaw(0, pTab->nCol+1);
                if (b == null)
                {
                    ctx.MallocFailed = true;
                    return;
                }
                for (int i = 0; i < table.Cols.; i++)
                    b.Append(table.Cols[i].Affinity);
                //b.Append('\0');
                table.ColAff = b.ToString();
            }
            v.ChangeP4(-1, table.ColAff, Vdbe.P4T.TRANSIENT);
        }

        static bool ReadsTable(Parse p, int startAddr, int db, Table table)
        {
            Vdbe v = p.GetVdbe();
            int end = v.CurrentAddr();
#if !OMIT_VIRTUALTABLE
            VTable vtable = (E.IsVirtual(table) ? VTable.GetVTable(p.Ctx, table) : null);
#endif
            for (int i = startAddr; i < end; i++)
            {
                Vdbe.VdbeOp op = v.GetOp(i);
                Debug.Assert(op != null);
                if (op.Opcode == OP.OpenRead && op.P3 == db)
                {
                    int id = op.P2;
                    if (id == table.Id)
                        return true;
                    for (Index index = table.Index; index != null; index = index.Next)
                        if (id == index.Id)
                            return true;
                }
#if !OMIT_VIRTUALTABLE
                if (op.Opcode == OP.VOpen && op.P4.VTable == vtable)
                {
                    Debug.Assert(op.P4.VTable != null);
                    Debug.Assert(op.P4Type == Vdbe.P4T.VTAB);
                    return true;
                }
#endif
            }
            return false;
        }

#if !OMIT_AUTOINCREMENT
        static int AutoIncBegin(Parse parse, int db, Table table)
        {
            int memId = 0;      // Register holding maximum rowid
            if ((table.TabFlags & TF.Autoincrement) != 0)
            {
                Parse toplevel = E.Parse_Toplevel(parse);
                AutoincInfo info = toplevel.Ainc;
                while (info != null && info.Table != table) info = info.Next;
                if (info == null)
                {
                    info = new AutoincInfo();
                    if (info == null) return 0;
                    info.Next = toplevel.Ainc;
                    toplevel.Ainc = info;
                    info.Table = table;
                    info.DB = db;
                    toplevel.Mems++;                // Register to hold name of table
                    info.RegCtr = ++toplevel.Mems;  // Max rowid register
                    toplevel.Mems++;                // Rowid in sqlite_sequence
                }
                memId = info.RegCtr;
            }
            return memId;
        }

        public static void AutoincrementBegin(Parse parse)
        {
            Context ctx = parse.Ctx; // The database connection
            Vdbe v = parse.V; // VDBE under construction

            // This routine is never called during trigger-generation.  It is only called from the top-level
            Debug.Assert(parse.TriggerTab == null);
            Debug.Assert(parse == E.Parse_Toplevel(parse));

            Debug.Assert(v != null); // We failed long ago if this is not so
            for (AutoincInfo p = parse.Ainc; p != null; p = p.Next) // Information about an AUTOINCREMENT
            {
                Context.DB dbAsObj = ctx.DBs[p.DB]; // Database only autoinc table
                int memId = p.RegCtr; // Register holding max rowid
                Debug.Assert(Btree.SchemaMutexHeld(ctx, 0, dbAsObj.Schema));
                OpenTable(parse, 0, p.DB, dbAsObj.Schema.SeqTable, OP.OpenRead);
                int addr = v.CurrentAddr(); // A VDBE address
                v.AddOp4(OP.String8, 0, memId - 1, 0, p.Table.Name, 0);
                v.AddOp2(OP.Rewind, 0, addr + 9);
                v.AddOp3(OP.Column, 0, 0, memId);
                v.AddOp3(OP.Ne, memId - 1, addr + 7, memId);
                v.ChangeP5(AFF.BIT_JUMPIFNULL);
                v.AddOp2(OP.Rowid, 0, memId + 1);
                v.AddOp3(OP.Column, 0, 1, memId);
                v.AddOp2(OP.Goto, 0, addr + 9);
                v.AddOp2(OP.Next, 0, addr + 2);
                v.AddOp2(OP.Integer, 0, memId);
                v.AddOp0(OP.Close);
            }
        }

        static void AutoIncStep(Parse parse, int memId, int regRowid)
        {
            if (memId > 0)
                parse.V.AddOp2(OP.MemMax, memId, regRowid);
        }

        public static void AutoincrementEnd(Parse parse)
        {
            Vdbe v = parse.V;
            Context ctx = parse.Ctx;

            Debug.Assert(v != null);
            for (AutoincInfo p = parse.Ainc; p != null; p = p.Next)
            {
                Context.DB dbAsObj = ctx.DBs[p.DB];
                int memId = p.RegCtr;

                int rec = Expr.GetTempReg(parse);
                Debug.Assert(Btree.SchemaMutexHeld(ctx, 0, dbAsObj.Schema));
                OpenTable(parse, 0, p.DB, dbAsObj.Schema.SeqTable, OP.OpenWrite);
                int j1 = v.AddOp1(OP.NotNull, memId + 1);
                int j2 = v.AddOp0(OP.Rewind);
                int j3 = v.AddOp3(OP.Column, 0, 0, rec);
                int j4 = v.AddOp3(OP.Eq, memId - 1, 0, rec);
                v.AddOp2(OP.Next, 0, j3);
                v.JumpHere(j2);
                v.AddOp2(OP.NewRowid, 0, memId + 1);
                int j5 = v.AddOp0(OP.Goto);
                v.JumpHere(j4);
                v.AddOp2(OP.Rowid, 0, memId + 1);
                v.JumpHere(j1);
                v.JumpHere(j5);
                v.AddOp3(OP.MakeRecord, memId - 1, 2, rec);
                v.AddOp3(OP.Insert, 0, rec, memId + 1);
                v.ChangeP5(Vdbe.OPFLAG.APPEND);
                v.AddOp0(OP.Close);
                Expr.ReleaseTempReg(parse, rec);
            }
        }
        //#else
        //#define AutoIncBegin(A,B,C) (0)
        //#define AutoIncStep(A,B,C)
#endif

        public static RC CodeCoroutine(Parse parse, Select select, SelectDest dest)
        {
            int regYield = ++parse.Mems; // Register holding co-routine entry-point
            int regEof = ++parse.Mems; // Register holding co-routine completion flag
            Vdbe v = parse.GetVdbe(); // VDBE under construction
            int addrTop = v.CurrentAddr(); // Top of the co-routine
            v.AddOp2(OP.Integer, addrTop + 2, regYield);              // X <- A
            v.Comment("Co-routine entry point");
            v.AddOp2(OP.Integer, 0, regEof);                        // EOF <- 0
            v.Comment("Co-routine completion flag");
            Select.DestInit(dest, SRT.Coroutine, regYield);
            int j1 = v.AddOp2(OP.Goto, 0, 0);                       // Jump instruction
            RC rc = Select.Select_(parse, select, dest);
            Debug.Assert(parse.Errs == 0 || rc != 0);
            if (parse.Ctx.MallocFailed && rc == RC.OK) rc = RC.NOMEM;
            if (rc != 0) return rc;
            v.AddOp2(OP.Integer, 1, regEof);                        // EOF <- 1
            v.AddOp1(OP.Yield, regYield);                           // yield X
            v.AddOp2(OP.Halt, RC.INTERNAL, OE.Abort);
            v.Comment("End of coroutine");
            v.JumpHere(j1);                                         // label B:
            return rc;
        }

        // OVERLOADS, so I don't need to rewrite parse.c
        static void Insert_(Parse parse, SrcList tabList, int dummy1, int dummy2, IdList column, OE onError) { Insert_(parse, tabList, null, null, column, onError); }
        static void Insert_(Parse parse, SrcList tabList, int dummy1, Select select, IdList column, OE onError) { Insert_(parse, tabList, null, select, column, onError); }
        static void Insert_(Parse parse, SrcList tabList, ExprList list, int dummy1, IdList column, OE onError) { Insert_(parse, tabList, list, null, column, onError); }
        static void Insert_(Parse parse, SrcList tabList, ExprList list, Select select, IdList column, OE onError)
        {
            int i = 0, j = 0;
            int idx = 0;            // Loop counters
            int baseCur = 0;      // VDBE VdbeCursor number for pTab
            int keyColumn = -1;   // Column that is the INTEGER PRIMARY KEY

            Context ctx = parse.Ctx; // The main database structure
            SelectDest dest = new SelectDest(); // Destination for SELECT on rhs of INSERT //: _memset(&dest, 0, sizeof(dest));
            if (parse.Errs != 0 || ctx.MallocFailed)
                goto insert_cleanup;

            // Locate the table into which we will be inserting new information.
            Debug.Assert(tabList.Srcs == 1);
            string tableName = tabList.Ids[0].Name; // Name of the table into which we are inserting
            if (C._NEVER(tableName == null))
                goto insert_cleanup;
            Table table = Delete.SrcListLookup(parse, tabList); // The table to insert into.  aka TABLE
            if (table == null)
                goto insert_cleanup;
            int db = Prepare.SchemaToIndex(ctx, table.Schema); // Index of database holding TABLE
            Debug.Assert(db < ctx.DBs.length);
            Context.DB dbAsObj = ctx.DBs[db]; // The database containing table being inserted into
#if !OMIT_AUTHORIZATION
            string dbName = dbAsObj.Name; // Name of the database holding this table
            if (Auth.Check(parse, AUTH.INSERT, table.Name, null, dbName) != 0)
                goto insert_cleanup;
#endif

#if !OMIT_TRIGGER
            // Figure out if we have any triggers and if the table being inserted into is a view
            TRIGGER tmask; // Mask of trigger times
            Trigger trigger = Trigger.TriggersExist(parse, table, TK.INSERT, null, out tmask); // List of triggers on pTab, if required
            Debug.Assert((trigger != null && tmask != 0) || (trigger == null && tmask == 0));
#if !OMIT_VIEW
            bool isView = (table.Select != null); // True if attempting to insert into a view
#else
            bool isView = false;
#endif
#else
            Trigger trigger = null;
            int tmask = 0;
            bool isView = false;
#endif

#if !OMIT_VIEW
            // If pTab is really a view, make sure it has been initialized. ViewGetColumnNames() is a no-op if pTab is not a view (or virtual module table).
            if (parse.ViewGetColumnNames(table) != -0)
                goto insert_cleanup;
#endif

            // Ensure that: (a) the table is not read-only,  (b) that if it is a view then ON INSERT triggers exist
            if (Delete.IsReadOnly(parse, table, (tmask != 0)))
                goto insert_cleanup;

            // Allocate a VDBE
            Vdbe v = parse.GetVdbe(); // Generate code into this virtual machine
            if (v == null) goto insert_cleanup;
            if (parse.Nested == 0) v.CountChanges();
            parse.BeginWriteOperation((select != null || trigger != null ? 1 : 0), db);

#if !OMIT_XFER_OPT
            // If the statement is of the form
            //
            //       INSERT INTO <table1> SELECT * FROM <table2>;
            //
            // Then special optimizations can be applied that make the transfer very fast and which reduce fragmentation of indices.
            //
            // This is the 2nd template.
            if (column == null && XferOptimization(parse, table, select, onError, db))
            {
                Debug.Assert(trigger == null);
                Debug.Assert(list == null);
                goto insert_end;
            }
#endif

            // Register allocations
            int regFromSelect = 0;  // Base register for data coming from SELECT
            int regAutoinc = 0;   // Register holding the AUTOINCREMENT counter
            int regRowCount = 0;  // Memory cell used for the row counter
            int regIns;           // Block of regs holding rowid+data being inserted
            int regRowid;         // registers holding insert rowid
            int regData;          // register holding first column to insert
            int regEof = 0;       // Register recording end of SELECT data
            int[] regIdxs = null; // One register allocated to each index

            // If this is an AUTOINCREMENT table, look up the sequence number in the sqlite_sequence table and store it in memory cell regAutoinc.
            regAutoinc = AutoIncBegin(parse, db, table);

            // Figure out how many columns of data are supplied.  If the data is coming from a SELECT statement, then generate a co-routine that
            // produces a single row of the SELECT on each invocation.  The co-routine is the common header to the 3rd and 4th templates.
            int columns; // Number of columns in the data
            bool useTempTable = false; // Store SELECT results in intermediate table
            int srcTab = 0; // Data comes from this temporary cursor if >=0
            int addrSelect = 0; // Address of coroutine that implements the SELECT
            if (select != null)
            {
                // Data is coming from a SELECT.  Generate a co-routine to run that SELECT.
                RC rc = CodeCoroutine(parse, select, dest);
                if (rc != 0) goto insert_cleanup;

                regEof = dest.SDParmId + 1;
                regFromSelect = dest.SdstId;
                Debug.Assert(select.EList != null);
                columns = select.EList.Exprs;
                Debug.Assert(dest.SdstId == columns);

                // Set useTempTable to TRUE if the result of the SELECT statement should be written into a temporary table (template 4).  Set to
                // FALSE if each* row of the SELECT can be written directly into the destination table (template 3).
                //
                // A temp table must be used if the table being updated is also one of the tables being read by the SELECT statement.  Also use a 
                // temp table in the case of row triggers.
                if (trigger != null || ReadsTable(parse, addrSelect, db, table))
                    useTempTable = true;

                if (useTempTable)
                {
                    // Invoke the coroutine to extract information from the SELECT and add it to a transient table srcTab.  The code generated
                    // here is from the 4th template:
                    //
                    //      B: open temp table
                    //      L: yield X
                    //         if EOF goto M
                    //         insert row from R..R+n into temp table
                    //         goto L
                    //      M: ...
                    srcTab = parse.Tabs++;
                    int regRec = Expr.GetTempReg(parse);                // Register to hold packed record
                    int regTempRowid = Expr.GetTempReg(parse);          // Register to hold temp table ROWID
                    v.AddOp2(OP.OpenEphemeral, srcTab, columns);
                    int addrTop = v.AddOp1(OP.Yield, dest.SDParmId);    // Label "L"
                    int addrIf = v.AddOp1(OP.If, regEof);               // Address of jump to M
                    v.AddOp3(OP.MakeRecord, regFromSelect, columns, regRec);
                    v.AddOp2(OP.NewRowid, srcTab, regTempRowid);
                    v.AddOp3(OP.Insert, srcTab, regRec, regTempRowid);
                    v.AddOp2(OP.Goto, 0, addrTop);
                    v.JumpHere(addrIf);
                    Expr.ReleaseTempReg(parse, regRec);
                    Expr.ReleaseTempReg(parse, regTempRowid);
                }
            }
            else
            {
                // This is the case if the data for the INSERT is coming from a VALUES clause
                NameContext sNC = new NameContext();
                sNC.Parse = parse;
                srcTab = -1;
                Debug.Assert(!useTempTable);
                columns = (list != null ? list.Exprs : 0);
                for (i = 0; i < columns; i++)
                    if (Resolve.ExprNames(sNC, ref list.Ids[i].Expr))
                        goto insert_cleanup;
            }

            // Make sure the number of columns in the source data matches the number of columns to be inserted into the table.
            int hidden = 0; // Number of hidden columns if TABLE is virtual
            if (E.IsVirtual(table))
                for (i = 0; i < table.Cols.length; i++)
                    hidden += (E.IsHiddenColumn(table.Cols[i]) ? 1 : 0);
            if (column == null && columns != 0 && columns != (table.Cols.length - hidden))
            {
                parse.ErrorMsg("table %S has %d columns but %d values were supplied", tabList, 0, table.Cols.length - hidden, columns);
                goto insert_cleanup;
            }
            if (column != null && columns != column.Ids.length)
            {
                parse.ErrorMsg("%d values for %d columns", columns, column.Ids.length);
                goto insert_cleanup;
            }

            // If the INSERT statement included an IDLIST term, then make sure all elements of the IDLIST really are columns of the table and 
            // remember the column indices.
            //
            // If the table has an INTEGER PRIMARY KEY column and that column is named in the IDLIST, then record in the keyColumn variable
            // the index into IDLIST of the primary key column.  keyColumn is the index of the primary key as it appears in IDLIST, not as
            // is appears in the original table.  (The index of the primary key in the original table is table->iPKey.)
            if (column != null)
            {
                for (i = 0; i < column.Ids.length; i++)
                    column.Ids[i].Idx = -1;
                for (i = 0; i < column.Ids.length; i++)
                {
                    for (j = 0; j < table.Cols.length; j++)
                    {
                        if (string.Equals(column.Ids[i].Name, table.Cols[j].Name, StringComparison.OrdinalIgnoreCase))
                        {
                            column.Ids[i].Idx = j;
                            if (j == table.PKey)
                                keyColumn = i;
                            break;
                        }
                    }
                    if (j >= table.Cols.length)
                    {
                        if (Expr.IsRowid(column.Ids[i].Name))
                            keyColumn = i;
                        else
                        {
                            parse.ErrorMsg("table %S has no column named %s", tabList, 0, column.Ids[i].Name);
                            parse.CheckSchema = 1;
                            goto insert_cleanup;
                        }
                    }
                }
            }

            // If there is no IDLIST term but the table has an integer primary key, the set the keyColumn variable to the primary key column index
            // in the original table definition.
            if (column == null && columns > 0)
                keyColumn = table.PKey;

            // Initialize the count of rows to be inserted
            if ((ctx.Flags & Context.FLAG.CountRows) != 0)
            {
                regRowCount = ++parse.Mems;
                v.AddOp2(OP.Integer, 0, regRowCount);
            }

            // If this is not a view, open the table and and all indices
            if (!isView)
            {
                baseCur = parse.Tabs;
                int idxs = OpenTableAndIndices(parse, table, baseCur, OP.OpenWrite);
                regIdxs = new int[idxs + 1]; //: _tagalloc(ctx, sizeof(int)*(idxs+1));
                if (regIdxs == null)
                    goto insert_cleanup;
                for (i = 0; i < idxs; i++)
                    regIdxs[i] = ++parse.Mems;
            }

            // This is the top of the main insertion loop
            int addrInsTop = 0; // Jump to label "D"
            int addrCont = 0; // Top of insert loop. Label "C" in templates 3 and 4
            if (useTempTable)
            {
                // This block codes the top of loop only.  The complete loop is the following pseudocode (template 4):
                //
                //         rewind temp table
                //      C: loop over rows of intermediate table
                //           transfer values form intermediate table into <table>
                //         end loop
                //      D: ...
                addrInsTop = v.AddOp1(OP.Rewind, srcTab);
                addrCont = v.CurrentAddr();
            }
            else if (select != null)
            {
                // This block codes the top of loop only.  The complete loop is the following pseudocode (template 3):
                //
                //      C: yield X
                //         if EOF goto D
                //         insert the select result into <table> from R..R+n
                //         goto C
                //      D: ...
                addrCont = v.AddOp1(OP.Yield, dest.SDParmId);
                addrInsTop = v.AddOp1(OP.If, regEof);
            }

            // Allocate registers for holding the rowid of the new row, the content of the new row, and the assemblied row record.
            regRowid = regIns = parse.Mems + 1;
            parse.Mems += table.Cols.length + 1;
            if (E.IsVirtual(table))
            {
                regRowid++;
                parse.Mems++;
            }
            regData = regRowid + 1;

            // Run the BEFORE and INSTEAD OF triggers, if there are any
            int endOfLoop = v.MakeLabel(); // Label for the end of the insertion loop
#if !OMIT_TRIGGER
            if ((tmask & TRIGGER.BEFORE) != 0)
            {
                int regCols = Expr.GetTempRange(parse, table.Cols.length + 1);

                // build the NEW.* reference row.  Note that if there is an INTEGER PRIMARY KEY into which a NULL is being inserted, that NULL will be
                // translated into a unique ID for the row.  But on a BEFORE trigger, we do not know what the unique ID will be (because the insert has
                // not happened yet) so we substitute a rowid of -1
                if (keyColumn < 0)
                    v.AddOp2(OP.Integer, -1, regCols);
                else
                {
                    if (useTempTable)
                        v.AddOp3(OP.Column, srcTab, keyColumn, regCols);
                    else
                    {
                        Debug.Assert(select == null); // Otherwise useTempTable is true
                        Expr.Code(parse, list.Ids[keyColumn].Expr, regCols);
                    }
                    int j1 = v.AddOp1(OP.NotNull, regCols);
                    v.AddOp2(OP.Integer, -1, regCols);
                    v.JumpHere(j1);
                    v.AddOp1(OP.MustBeInt, regCols);
                }

                // Cannot have triggers on a virtual table. If it were possible, this block would have to account for hidden column.
                Debug.Assert(!E.IsVirtual(table));

                // Create the new column data
                for (i = 0; i < table.Cols.length; i++)
                {
                    if (column == null)
                        j = i;
                    else
                    {
                        for (j = 0; j < column.Ids.length; j++)
                            if (column.Ids[j].Idx == i)
                                break;
                    }
                    if ((!useTempTable && null == list) || (column != null && j >= column.Ids.length))
                        Expr.Code(parse, table.Cols[i].Dflt, regCols + i + 1);
                    else if (useTempTable)
                        v.AddOp3(OP.Column, srcTab, j, regCols + i + 1);
                    else
                    {
                        Debug.Assert(select == null); // Otherwise useTempTable is true
                        Expr.CodeAndCache(parse, list.Ids[j].Expr, regCols + i + 1);
                    }
                }

                // If this is an INSERT on a view with an INSTEAD OF INSERT trigger, do not attempt any conversions before assembling the record.
                // If this is a real table, attempt conversions as required by the table column affinities.
                if (!isView)
                {
                    v.AddOp2(OP.Affinity, regCols + 1, table.Cols.length);
                    TableAffinityStr(v, table);
                }

                // Fire BEFORE or INSTEAD OF triggers
                trigger.CodeRowTrigger(parse, TK.INSERT, null, TRIGGER.BEFORE, table, regCols - table.Cols.length - 1, onError, endOfLoop);

                Expr.ReleaseTempRange(parse, regCols, table.Cols.length + 1);
            }
#endif

            // Push the record number for the new entry onto the stack.  The record number is a randomly generate integer created by NewRowid
            // except when the table has an INTEGER PRIMARY KEY column, in which case the record number is the same as that column.
            bool appendFlag = false; // True if the insert is likely to be an append
            if (!isView)
            {
                if (E.IsVirtual(table))
                    v.AddOp2(OP.Null, 0, regIns); // The row that the VUpdate opcode will delete: none
                if (keyColumn >= 0)
                {
                    if (useTempTable)
                        v.AddOp3(OP.Column, srcTab, keyColumn, regRowid);
                    else if (select != null)
                        v.AddOp2(OP.SCopy, regFromSelect + keyColumn, regRowid);
                    else
                    {
                        Expr.Code(parse, list.Ids[keyColumn].Expr, regRowid);
                        Vdbe.VdbeOp op = v.GetOp(-1);
                        if (C._ALWAYS(op != null) && op.Opcode == OP.Null && !E.IsVirtual(table))
                        {
                            appendFlag = true;
                            op.Opcode = OP.NewRowid;
                            op.P1 = baseCur;
                            op.P2 = regRowid;
                            op.P3 = regAutoinc;
                        }
                    }
                    // If the PRIMARY KEY expression is NULL, then use OP_NewRowid to generate a unique primary key value.
                    if (!appendFlag)
                    {
                        int j1;
                        if (!E.IsVirtual(table))
                        {
                            j1 = v.AddOp1(OP.NotNull, regRowid);
                            v.AddOp3(OP.NewRowid, baseCur, regRowid, regAutoinc);
                            v.JumpHere(j1);
                        }
                        else
                        {
                            j1 = v.CurrentAddr();
                            v.AddOp2(OP.IsNull, regRowid, j1 + 2);
                        }
                        v.AddOp1(OP.MustBeInt, regRowid);
                    }
                }
                else if (E.IsVirtual(table))
                    v.AddOp2(OP.Null, 0, regRowid);
                else
                {
                    v.AddOp3(OP.NewRowid, baseCur, regRowid, regAutoinc);
                    appendFlag = true;
                }
                AutoIncStep(parse, regAutoinc, regRowid);

                // Push onto the stack, data for all columns of the new entry, beginning with the first column.
                hidden = 0;
                for (i = 0; i < table.Cols.length; i++)
                {
                    int regStoreId = regRowid + 1 + i;
                    if (i == table.PKey)
                    {
                        // The value of the INTEGER PRIMARY KEY column is always a NULL. Whenever this column is read, the record number will be substituted
                        // in its place.  So will fill this column with a NULL to avoid taking up data space with information that will never be used.
                        v.AddOp2(OP.Null, 0, regStoreId);
                        continue;
                    }
                    if (column == null)
                    {
                        if (E.IsHiddenColumn(table.Cols[i]))
                        {
                            Debug.Assert(E.IsVirtual(table));
                            j = -1;
                            hidden++;
                        }
                        else
                            j = i - hidden;
                    }
                    else
                    {
                        for (j = 0; j < column.Ids.length; j++)
                            if (column.Ids[j].Idx == i) break;
                    }
                    if (j < 0 || columns == 0 || (column != null && j >= column.Ids.length))
                        Expr.Code(parse, table.Cols[i].pDflt, regStoreId);
                    else if (useTempTable)
                        v.AddOp3(OP.Column, srcTab, j, regStoreId);
                    else if (select != null)
                        v.AddOp2(OP.SCopy, regFromSelect + j, regStoreId);
                    else
                        Expr.Code(parse, list.Ids[j].Expr, regStoreId);
                }

                // Generate code to check constraints and generate index keys and do the insertion.
#if !OMIT_VIRTUALTABLE
                if (E.IsVirtual(table))
                {
                    VTable vtable = VTable.GetVTable(ctx, table);
                    VTable.MakeWritable(parse, table);
                    v.AddOp4(OP.VUpdate, 1, table.Cols.length + 2, regIns, vtable, Vdbe.P4T.VTAB);
                    v.ChangeP5((byte)(onError == OE.Default ? OE.Abort : onError));
                    parse.MayAbort();
                }
                else
#endif
                {
                    bool isReplace; // Set to true if constraints may cause a replace
                    GenerateConstraintChecks(parse, table, baseCur, regIns, regIdxs, keyColumn >= 0 ? 1 : 0, false, onError, endOfLoop, out isReplace);
                    parse.FKCheck(table, 0, regIns);
                    CompleteInsertion(parse, table, baseCur, regIns, regIdxs, false, appendFlag, !isReplace);
                }
            }

            // Update the count of rows that are inserted
            if ((ctx.Flags & Context.FLAG.CountRows) != 0)
                v.AddOp2(OP.AddImm, regRowCount, 1);

#if !OMIT_TRIGGER
            if (trigger != null)  // Code AFTER triggers
                trigger.CodeRowTrigger(parse, TK.INSERT, null, TRIGGER.AFTER, table, regData - 2 - table.Cols.length, onError, endOfLoop);
#endif

            // The bottom of the main insertion loop, if the data source is a SELECT statement.
            v.ResolveLabel(endOfLoop);
            if (useTempTable)
            {
                v.AddOp2(OP.Next, srcTab, addrCont);
                v.JumpHere(addrInsTop);
                v.AddOp1(OP.Close, srcTab);
            }
            else if (select != null)
            {
                v.AddOp2(OP.Goto, 0, addrCont);
                v.JumpHere(addrInsTop);
            }

            if (!E.IsVirtual(table) && !isView)
            {
                // Close all tables opened
                v.AddOp1(OP.Close, baseCur);
                Index index; // For looping over indices of the table
                for (idx = 1, index = table.Index; index != null; index = index.Next, idx++)
                    v.AddOp1(OP.Close, idx + baseCur);
            }

        insert_end:
            // Update the sqlite_sequence table by storing the content of the maximum rowid counter values recorded while inserting into autoincrement tables.
            if (parse.Nested == 0 && parse.TriggerTab == null)
                AutoincrementEnd(parse);

            // Return the number of rows inserted. If this routine is generating code because of a call to sqlite3NestedParse(), do not invoke the callback function.
            if ((ctx.Flags & Context.FLAG.CountRows) != 0 && parse.Nested == 0 && parse.TriggerTab == null)
            {
                v.AddOp2(OP.ResultRow, regRowCount, 1);
                v.SetNumCols(1);
                v.SetColName(0, COLNAME_NAME, "rows inserted", C.DESTRUCTOR_STATIC);
            }

        insert_cleanup:
            Expr.SrcListDelete(ctx, ref tabList);
            Expr.ListDelete(ctx, ref list);
            Select.Delete(ctx, ref select);
            Expr.IdListDelete(ctx, ref column);
            C._tagfree(ctx, ref regIdxs);
        }

        public static void GenerateConstraintChecks(Parse parse, Table table, int baseCur, int regRowid, int[] regIdxs, int rowidChng, bool isUpdate, OE overrideError, int ignoreDest, out bool mayReplaceOut)
        {
            Context ctx = parse.Ctx; // Database connection
            Vdbe v = parse.GetVdbe(); // VDBE under constrution
            Debug.Assert(v != null);
            Debug.Assert(table.Select == null); // This table is not a VIEW
            int colsLength = table.Cols.length; // Number of columns
            int regData = regRowid + 1; // Register containing first data column
            OE onError; // Conflict resolution strategy

            // Test all NOT NULL constraints.
            int i;
            for (i = 0; i < colsLength; i++)
            {
                if (i == table.PKey)
                    continue;
                onError = table.Cols[i].NotNull;
                if (onError == OE.None) continue;
                if (overrideError != OE.Default) onError = overrideError;
                else if (onError == OE.Default) onError = OE.Abort;
                if (onError == OE.Replace && table.Cols[i].Dflt == null) onError = OE.Abort;
                Debug.Assert(onError == OE.Rollback || onError == OE.Abort || onError == OE.Fail || onError == OE.Ignore || onError == OE.Replace);
                switch (onError)
                {
                    case OE.Abort:
                        {
                            parse.MayAbort();
                            goto case OE.Fail;
                        }
                    case OE.Rollback:
                    case OE.Fail:
                        {
                            v.AddOp3(OP.HaltIfNull, RC.RC_CONSTRAINT_NOTNULL, onError, regData + i);
                            string msg = C._mtagprintf(parse.Ctx, "%s.%s may not be NULL", table.Name, table.Cols[i].Name);
                            v.ChangeP4(-1, msg, Vdbe.P4T.DYNAMIC);
                            break;
                        }
                    case OE.Ignore:
                        {
                            v.AddOp2(OP.IsNull, regData + i, ignoreDest);
                            break;
                        }
                    default:
                        {
                            Debug.Assert(onError == OE.Replace);
                            int j1 = v.AddOp1(OP.NotNull, regData + i); // Addresss of jump instruction
                            Expr.Code(parse, table.Cols[i].Dflt, regData + i);
                            v.JumpHere(j1);
                            break;
                        }
                }
            }

#if !OMIT_CHECK
            // Test all CHECK constraints
            if (table.Check != null && (ctx.Flags & Context.FLAG.IgnoreChecks) == 0)
            {
                ExprList check = table.Check;
                parse.CkBase = regData;
                onError = (overrideError != OE.Default ? overrideError : OE.Abort);
                for (i = 0; i < check.Exprs; i++)
                {
                    int allOk = v.MakeLabel();
                    check.Ids[i].Expr.IfTrue(parse, allOk, AFF.BIT_JUMPIFNULL);
                    if (onError == OE.Ignore)
                        v.AddOp2(OP.Goto, 0, ignoreDest);
                    else
                    {
                        if (onError == OE.Replace) onError = OE.Abort; // IMP: R-15569-63625
                        string consName = check.Ids[i].Name;
                        consName = (consName != null ? C._mtagprintf(ctx, "constraint %s failed", consName) : null);
                        parse.HaltConstraint(RC.CONSTRAINT_CHECK, onError, consName, 0);
                    }
                    v.ResolveLabel(allOk);
                }
            }
#endif

            // If we have an INTEGER PRIMARY KEY, make sure the primary key of the new record does not previously exist.  Except, if this
            // is an UPDATE and the primary key is not changing, that is OK.
            bool seenReplace = false; // True if REPLACE is used to resolve INT PK conflict
            int j2 = 0, j3; // Addresses of jump instructions
            if (rowidChng != 0)
            {
                onError = table.KeyConf;
                if (overrideError != OE.Default) onError = overrideError;
                else if (onError == OE.Default) onError = OE.Abort;
                if (isUpdate)
                    j2 = v.AddOp3(OP.Eq, regRowid, 0, rowidChng);
                j3 = v.AddOp3(OP.NotExists, baseCur, 0, regRowid);
                switch (onError)
                {
                    default:
                        {
                            onError = OE.Abort;
                        }
                        goto case OE.Rollback; // Fall thru into the next case
                    case OE.Rollback:
                    case OE.Abort:
                    case OE.Fail:
                        {
                            parse.HaltConstraint(RC.CONSTRAINT_PRIMARYKEY, onError, "PRIMARY KEY must be unique", Vdbe.P4T.STATIC);
                            break;
                        }
                    case OE.Replace:
                        {
                            // If there are DELETE triggers on this table and the recursive-triggers flag is set, call GenerateRowDelete() to
                            // remove the conflicting row from the table. This will fire the triggers and remove both the table and index b-tree entries.
                            //
                            // Otherwise, if there are no triggers or the recursive-triggers flag is not set, but the table has one or more indexes, call 
                            // GenerateRowIndexDelete(). This removes the index b-tree entries only. The table b-tree entry will be replaced by the new entry 
                            // when it is inserted.  
                            //
                            // If either GenerateRowDelete() or GenerateRowIndexDelete() is called, also invoke MultiWrite() to indicate that this VDBE may require
                            // statement rollback (if the statement is aborted after the delete takes place). Earlier versions called sqlite3MultiWrite() regardless,
                            // but being more selective here allows statements like:
                            //
                            //   REPLACE INTO t(rowid) VALUES($newrowid)
                            //
                            // to run without a statement journal if there are no indexes on the table.
                            Trigger trigger = null;
                            int dummy1;
                            if ((ctx.Flags & Context.FLAG.RecTriggers) != 0)
                                trigger = sqlite3TriggersExist(parse, table, TK.DELETE, null, out dummy1);
                            if (trigger != null || parse.FKRequired(table, null, 0))
                            {
                                parse.MultiWrite();
                                sqlite3GenerateRowDelete(parse, table, baseCur, regRowid, 0, trigger, OE.Replace);
                            }
                            else if (table.Index != null)
                            {
                                parse.MultiWrite();
                                sqlite3GenerateRowIndexDelete(parse, table, baseCur, 0);
                            }
                            seenReplace = true;
                            break;
                        }
                    case OE.Ignore:
                        {
                            Debug.Assert(!seenReplace);
                            v.AddOp2(OP.Goto, 0, ignoreDest);
                            break;
                        }
                }
                v.JumpHere(j3);
                if (isUpdate)
                    v.JumpHere(j2);
            }

            // Test all UNIQUE constraints by creating entries for each UNIQUE index and making sure that duplicate entries do not already exist.
            // Add the new records to the indices as we go.
            int curId; // Table cursor number
            Index index; // Pointer to one of the indices
            int regOldRowid = (rowidChng != 0 && isUpdate ? rowidChng : regRowid);
            for (curId = 0, index = table.Index; index != null; index = index.Next, curId++)
            {
                if (regIdxs[curId] == 0) continue;  // Skip unused indices

                // Create a key for accessing the index entry
                int regIdx = Expr.GetTempRange(parse, index.Columns.length + 1);
                for (i = 0; i < index.Columns.length; i++)
                {
                    int idx = index.Columns[i];
                    if (idx == table.PKey)
                        v.AddOp2(OP.SCopy, regRowid, regIdx + i);
                    else
                        v.AddOp2(OP.SCopy, regData + idx, regIdx + i);
                }
                v.AddOp2(OP.SCopy, regRowid, regIdx + i);
                v.AddOp3(OP.MakeRecord, regIdx, index.Columns.length + 1, regIdxs[curId]);
                v.ChangeP4(-1, IndexAffinityStr(v, index), Vdbe.P4T.TRANSIENT);
                Expr.CacheAffinityChange(parse, regIdx, index.Columns.length + 1);

                // Find out what action to take in case there is an indexing conflict
                onError = index.OnError;
                if (onError == OE.None)
                {
                    Expr.ReleaseTempRange(parse, regIdx, index.Columns.length + 1);
                    continue;  // pIdx is not a UNIQUE index
                }
                if (overrideError != OE.Default) onError = overrideError;
                else if (onError == OE.Default) onError = OE.Abort;
                if (seenReplace)
                {
                    if (onError == OE.Ignore) onError = OE.Replace;
                    else if (onError == OE.Fail) onError = OE.Abort;
                }

                // Check to see if the new index entry will be unique
                int regR = Expr.GetTempReg(parse);
                v.AddOp2(OP.SCopy, regOldRowid, regR);
                j3 = v.AddOp4(OP.IsUnique, baseCur + curId + 1, 0, regR, regIdx, Vdbe.P4T.INT32);
                Expr.ReleaseTempRange(parse, regIdx, index.Columns.length + 1);

                // Generate code that executes if the new index entry is not unique
                Debug.Assert(onError == OE.Rollback || onError == OE.Abort || onError == OE.Fail || onError == OE.Ignore || onError == OE.Replace);
                switch (onError)
                {
                    case OE.Rollback:
                    case OE.Abort:
                    case OE.Fail:
                        {
                            TextBuilder b = new TextBuilder(200);
                            TextBuilder.Init(b, 0, 200);
                            b.Tag = parse.Ctx;
                            string sepText = (index.Columns.length > 1 ? "columns " : "column ");
                            for (int j = 0; j < index.Columns.length; j++)
                            {
                                string colName = table.Cols[index.Columns[j]].Name;
                                b.Append(sepText, -1);
                                sepText = ", ";
                                b.Append(colName, -1);
                            }
                            b.Append((index.Columns.length > 1 ? " are not unique" : " is not unique"), -1);
                            string err = b.ToString();
                            parse.HaltConstraint(RC.CONSTRAINT_UNIQUE, onError, err, 0);
                            C._tagfree(ctx, ref err);
                            break;
                        }
                    case OE.Ignore:
                        {
                            Debug.Assert(!seenReplace);
                            v.AddOp2(OP.Goto, 0, ignoreDest);
                            break;
                        }
                    default:
                        {
                            Trigger trigger = null;
                            Debug.Assert(onError == OE.Replace);
                            parse.MultiWrite();
                            int dummy1;
                            if ((ctx.Flags & Context.FLAG.RecTriggers) != 0)
                                trigger = Trigger.TriggersExist(parse, table, TK.DELETE, null, out dummy1);
                            Delete.GenerateRowDelete(parse, table, baseCur, regR, 0, trigger, OE.Replace);
                            seenReplace = true;
                            break;
                        }
                }
                v.JumpHere(j3);
                Expr.ReleaseTempReg(parse, regR);
            }
            mayReplaceOut = seenReplace;
        }

        public static void CompleteInsertion(Parse parse, Table table, int baseCur, int regRowid, int[] regIdxs, bool isUpdate, bool appendBias, bool useSeekResult)
        {
            Vdbe v = parse.GetVdbe();
            Debug.Assert(v != null);
            Debug.Assert(table.Select == null); // This table is not a VIEW
            int indexsLength;
            Index index;
            for (indexsLength = 0, index = table.Index; index != null; index = index.Next, indexsLength++) { }
            for (int i = indexsLength - 1; i >= 0; i--)
            {
                if (regIdxs[i] == 0) continue;
                v.AddOp2(OP.IdxInsert, baseCur + i + 1, regIdxs[i]);
                if (useSeekResult)
                    v.ChangeP5(Vdbe.OPFLAG.USESEEKRESULT);
            }
            int regData = regRowid + 1;
            int regRec = Expr.GetTempReg(parse);
            v.AddOp3(OP.MakeRecord, regData, table.Cols.length, regRec);
            TableAffinityStr(v, table);
            Expr.CacheAffinityChange(parse, regData, table.Cols.length);
            Vdbe.OPFLAG pik_flags;
            if (parse.Nested != 0)
                pik_flags = 0;
            else
            {
                pik_flags = Vdbe.OPFLAG.NCHANGE;
                pik_flags |= (isUpdate ? Vdbe.OPFLAG.ISUPDATE : Vdbe.OPFLAG.LASTROWID);
            }
            if (appendBias) pik_flags |= Vdbe.OPFLAG.APPEND;
            if (useSeekResult) pik_flags |= Vdbe.OPFLAG.USESEEKRESULT;
            v.AddOp3(OP.Insert, baseCur, regRec, regRowid);
            if (parse.Nested == 0)
                v.ChangeP4(-1, table.Name, Vdbe.P4T.TRANSIENT);
            v.ChangeP5(pik_flags);
        }

        public static int OpenTableAndIndices(Parse parse, Table table, int baseCur, OP op)
        {
            if (E.IsVirtual(table)) return 0;
            int db = Prepare.SchemaToIndex(parse.Ctx, table.Schema);
            Vdbe v = parse.GetVdbe();
            Debug.Assert(v != null);
            OpenTable(parse, baseCur, db, table, op);
            int i;
            Index index;
            for (i = 1, index = table.Index; index != null; index = index.Next, i++)
            {
                KeyInfo pKey = parse.IndexKeyinfo(index);
                Debug.Assert(index.Schema == table.Schema);
                v.AddOp4(op, i + baseCur, index.Id, db, key, Vdbe.P4T.KEYINFO_HANDOFF);
                v.Comment("%s", index.Name);
            }
            if (parse.Tabs < baseCur + i)
                parse.Tabs = baseCur + i;
            return i - 1;
        }

#if TEST
        static int _xferopt_count = 0;
#endif
#if !OMIT_XFER_OPT
        static bool XferCompatibleCollation(string z1, string z2)
        {
            if (z1 == null) return (z2 == null);
            if (z2 == null) return false;
            return string.Equals(z1, z2, StringComparison.OrdinalIgnoreCase);
        }

        static bool XferCompatibleIndex(Index dest, Index src)
        {
            Debug.Assert(dest != null && src != null);
            Debug.Assert(dest.Table != src.Table);
            if (dest.Columns.length != src.Columns.length) return false;    // Different number of columns
            if (dest.OnError != src.OnError) return false;                  // Different conflict resolution strategies
            for (int i = 0; i < src.Columns.length; i++)
            {
                if (src.Columns[i] != dest.Columns[i]) return false;      // Different columns indexed
                if (src.SortOrders[i] != dest.SortOrders[i]) return false;  // Different sort orders
                if (!XferCompatibleCollation(src.CollNames[i], dest.CollNames[i])) return false; // Different collating sequences
            }
            // If no test above fails then the indices must be compatible
            return true;
        }

        static bool XferOptimization(Parse parse, Table dest, Select select, OE onError, int dbDestId)
        {
            if (select == null) return false;                                   // Must be of the form  INSERT INTO ... SELECT ...
            if (sqlite3TriggerList(parse, dest) != null) return false;          // tab1 must not have triggers
#if !OMIT_VIRTUALTABLE
            if ((dest.TabFlags & TF.Virtual) != 0) return false;				// tab1 must not be a virtual table
#endif
            if (onError == OE.Default)
            {
                if (dest.PKey >= 0) onError = dest.KeyConf;
                if (onError == OE.Default) onError = OE.Abort;
            }
            Debug.Assert(select.Src != null);                                   // allocated even if there is no FROM clause
            if (select.Src.Srcs != 1) return false;                             // FROM clause must have exactly one term
            if (select.Src.Ids[0].Select != null) return false;                 // FROM clause cannot contain a subquery
            if (select.Where != null) return false;                             // SELECT may not have a WHERE clause
            if (select.OrderBy != null) return false;                           // SELECT may not have an ORDER BY clause
            // Do not need to test for a HAVING clause.  If HAVING is present but there is no ORDER BY, we will get an error.
            if (select.GroupBy != null) return false;                           // SELECT may not have a GROUP BY clause
            if (select.Limit != null) return false;                             // SELECT may not have a LIMIT clause
            Debug.Assert(select.Offset == null);                                // Must be so if pLimit==0
            if (select.Prior != null) return false;                             // SELECT may not be a compound query
            if ((select.SelFlags & SF.Distinct) != 0) return false;             // SELECT may not be DISTINCT
            ExprList list = select.EList;                                      // The result set of the SELECT
            Debug.Assert(list != null);
            if (list.Exprs != 1) return false;                                 // The result set must have exactly one column
            Debug.Assert(list.Ids[0].Expr != null);
            if (list.Ids[0].Expr.OP != TK.ALL) return false;                   // The result set must be the special operator "*"

            // At this point we have established that the statement is of the correct syntactic form to participate in this optimization.  Now
            // we have to check the semantics.
            SrcList.SrcListItem item = select.Src.Ids[0];                       // An element of pSelect.pSrc
            Table src = parse.LocateTableItem(false, item);                     // The table in the FROM clause of SELECT
            if (src == null) return false;                                      // FROM clause does not contain a real table
            if (src == dest) return false;                                      // tab1 and tab2 may not be the same table
#if !OMIT_VIRTUALTABLE
            if ((src.TabFlags & TF.Virtual) != 0) return false;                 // tab2 must not be a virtual table
#endif
            if (src.Select != null) return false;                               // tab2 may not be a view
            if (dest.Cols.length != src.Cols.length) return false;              // Number of columns must be the same in tab1 and tab2
            if (dest.PKey != src.PKey) return false;                            // Both tables must have the same INTEGER PRIMARY KEY
            for (int i = 0; i < dest.Cols.length; i++)
            {
                if (dest.Cols[i].Affinity != src.Cols[i].Affinity) return false; // Affinity must be the same on all columns
                if (!XferCompatibleCollation(dest.Cols[i].Coll, src.Cols[i].Coll)) return false; // Collating sequence must be the same on all columns
                if (dest.Cols[i].NotNull != 0 && src.Cols[i].NotNull == 0) return false; // tab2 must be NOT NULL if tab1 is
            }
            bool destHasUniqueIdx = false;   // True if pDest has a UNIQUE index
            Index srcIdx, destIdx; // Source and destination indices
            for (destIdx = dest.Index; destIdx != null; destIdx = destIdx.Next)
            {
                if (destIdx.OnError != OE.None) destHasUniqueIdx = true;
                for (srcIdx = src.Index; srcIdx != null; srcIdx = srcIdx.Next)
                    if (XferCompatibleIndex(destIdx, srcIdx)) break;
                if (srcIdx == null) return false;                               // pDestIdx has no corresponding index in pSrc
            }
#if !OMIT_CHECK
            if (dest.Check != null && Expr.ListCompare(src.Check, dest.Check) != 0) return false; // Tables have different CHECK constraints.  Ticket #2252
#endif
#if !OMIT_FOREIGN_KEY
            // Disallow the transfer optimization if the destination table constains any foreign key constraints.  This is more restrictive than necessary.
            // But the main beneficiary of the transfer optimization is the VACUUM command, and the VACUUM command disables foreign key constraints.  So
            // the extra complication to make this rule less restrictive is probably not worth the effort.  Ticket [6284df89debdfa61db8073e062908af0c9b6118e]
            if ((parse.Ctx.Flags & Context.FLAG.ForeignKeys) != 0 && dest.FKeys != null) return false;
#endif
            if ((parse.Ctx.Flags & Context.FLAG.CountRows) != 0) return false;  // xfer opt does not play well with PRAGMA count_changes

            // If we get this far, it means that the xfer optimization is at least a possibility, though it might only work if the destination
            // table (tab1) is initially empty.
#if TEST
            _xferopt_count++;
#endif
            int dbSrcId = Prepare.SchemaToIndex(parse.Ctx, src.Schema); // The database of pSrc
            Vdbe v = parse.GetVdbe(); // The VDBE we are building
            parse.CodeVerifySchema(dbSrcId);
            int srcId = parse.Tabs++; // Cursors from source and destination 
            int destId = parse.Tabs++; // Cursors from source and destination 
            int regAutoinc = AutoIncBegin(parse, dbDestId, dest); // Memory register used by AUTOINC
            OpenTable(parse, destId, dbDestId, dest, OP.OpenWrite);
            int addr1; // Loop addresses
            int emptyDestTest; // Address of test for empty pDest
            if ((dest.PKey < 0 && dest.Index != null) ||		        // (1)
                destHasUniqueIdx ||									// (2)
                (onError != OE.Abort && onError != OE.Rollback))	    // (3)
            {
                // In some circumstances, we are able to run the xfer optimization only if the destination table is initially empty.  This code makes
                // that determination.  Conditions under which the destination must be empty:
                //
                // (1) There is no INTEGER PRIMARY KEY but there are indices. (If the destination is not initially empty, the rowid fields
                //     of index entries might need to change.)
                //
                // (2) The destination has a unique index.  (The xfer optimization is unable to test uniqueness.)
                //
                // (3) onError is something other than OE_Abort and OE_Rollback.
                addr1 = v.AddOp2(OP.Rewind, destId, 0);
                emptyDestTest = v.AddOp2(OP.Goto, 0, 0);
                v.JumpHere(addr1);
            }
            else
                emptyDestTest = 0;
            OpenTable(parse, srcId, dbSrcId, src, OP.OpenRead);
            int emptySrcTest = v.AddOp2(OP.Rewind, srcId, 0); // Address of test for empty pSrc
            int regData = Expr.GetTempReg(parse); // Registers holding data and rowid
            int regRowid = Expr.GetTempReg(parse);// Registers holding data and rowid
            if (dest.PKey >= 0)
            {
                addr1 = v.AddOp2(OP.Rowid, srcId, regRowid);
                int addr2 = v.AddOp3(OP.NotExists, destId, 0, regRowid); // Loop addresses
                sqlite3HaltConstraint(parse, onError, "PRIMARY KEY must be unique", Vdbe.P4T.STATIC);
                v.JumpHere(addr2);
                AutoIncStep(parse, regAutoinc, regRowid);
            }
            else if (dest.Index == null)
                addr1 = v.AddOp2(OP.NewRowid, destId, regRowid);
            else
            {
                addr1 = v.AddOp2(OP.Rowid, srcId, regRowid);
                Debug.Assert((dest.TabFlags & TF.Autoincrement) == 0);
            }
            v.AddOp2(OP.RowData, srcId, regData);
            v.AddOp3(OP.Insert, destId, regData, regRowid);
            v.ChangeP5(Vdbe.OPFLAG.NCHANGE | Vdbe.OPFLAG.LASTROWID | Vdbe.OPFLAG.APPEND);
            v.ChangeP4(-1, dest.Name, 0);
            v.AddOp2(OP.Next, srcId, addr1);
            KeyInfo key; // Key information for an index
            for (destIdx = dest.Index; destIdx != null; destIdx = destIdx.Next)
            {
                for (srcIdx = src.Index; srcIdx != null; srcIdx = srcIdx.Next)
                    if (XferCompatibleIndex(destIdx, srcIdx))
                        break;
                Debug.Assert(srcIdx != null);
                v.AddOp2(OP.Close, srcId, 0);
                v.AddOp2(OP.Close, destId, 0);
                key = parse.IndexKeyinfo(srcIdx);
                v.AddOp4(OP.OpenRead, srcId, srcIdx.Id, dbSrcId, key, Vdbe.P4T.KEYINFO_HANDOFF);
                v.Comment("%s", srcIdx.Name);
                key = parse.IndexKeyinfo(destIdx);
                v.AddOp4(OP.OpenWrite, destId, destIdx.Id, dbDestId, key, Vdbe.P4T.KEYINFO_HANDOFF);
                v.Comment("%s", destIdx.Name);
                addr1 = v.AddOp2(OP.Rewind, srcId, 0);
                v.AddOp2(OP.RowKey, srcId, regData);
                v.AddOp3(OP.IdxInsert, destId, regData, 1);
                v.AddOp2(OP.Next, srcId, addr1 + 1);
                v.JumpHere(addr1);
            }
            v.JumpHere(emptySrcTest);
            Expr.ReleaseTempReg(parse, regRowid);
            Expr.ReleaseTempReg(parse, regData);
            v.AddOp2(OP.Close, srcId, 0);
            v.AddOp2(OP.Close, destId, 0);
            if (emptyDestTest != 0)
            {
                v.AddOp2(OP.Halt, RC.OK, 0);
                v.JumpHere(emptyDestTest);
                v.AddOp2(OP.Close, destId, 0);
                return false;
            }
            return true;
        }

#endif
    }
}

