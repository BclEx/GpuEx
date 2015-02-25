using System;
using System.Diagnostics;
namespace Core.Command
{
    public partial class Update
    {
        public static void sqlite3ColumnDefault(Vdbe v, Table table, int i, int regId)
        {
            Debug.Assert(table != null);
            if (table.Select == null)
            {
                TEXTENCODE encode = Context.CTXENCODE(v.Ctx);
                Column col = table.Cols[i];
                v.VdbeComment("%s.%s", table.Name, col.Name);
                Debug.Assert(i < table.Cols.length);
                Mem value = new Mem();
                sqlite3ValueFromExpr(v.Ctx, col.Dflt, encode, col.Affinity, ref value);
                if (value != null)
                    v.ChangeP4(-1, value, Vdbe.P4T.MEM);
#if !OMIT_FLOATING_POINT
                if (regId >= 0 && table.Cols[i].Affinity == AFF.REAL)
                    v.AddOp1(OP.RealAffinity, regId);
#endif
            }
        }

        public static void sqlite3Update(Parse parse, SrcList tabList, ExprList changes, Expr where_, OE onError)
        {
            int i, j;                   // Loop counters

            AuthContext sContext = new AuthContext(); // The authorization context
            Context ctx = parse.Ctx; // The database structure
            if (parse.Errs != 0 || ctx.MallocFailed)
                goto update_cleanup;
            Debug.Assert(tabList.Srcs == 1);

            // Locate the table which we want to update.
            Table table = sqlite3SrcListLookup(parse, tabList); // The table to be updated
            if (table == null) goto update_cleanup;
            int db = sqlite3SchemaToIndex(ctx, table.Schema); // Database containing the table being updated

            // Figure out if we have any triggers and if the table being updated is a view.
#if !OMIT_TRIGGER
            int tmask = 0; // Mask of TRIGGER_BEFORE|TRIGGER_AFTER
            Trigger trigger = sqlite3TriggersExist(parse, table, TK.UPDATE, changes, out tmask); // List of triggers on pTab, if required
#if OMIT_VIEW
            const bool isView = false;
#else
            bool isView = (table.Select != null); // True when updating a view (INSTEAD OF trigger)
#endif
            Debug.Assert(trigger != null || tmask == 0);
#else
            const Trigger trigger = null;
            const int tmask = 0;
            const bool isView = false;
#endif

            if (sqlite3ViewGetColumnNames(parse, table) != 0 || sqlite3IsReadOnly(parse, table, tmask))
                goto update_cleanup;

            int[] xrefs = new int[table.Cols.length]; // xrefs[i] is the index in pChanges->a[] of the an expression for the i-th column of the table. xrefs[i]==-1 if the i-th column is not changed.
            if (xrefs == null) goto update_cleanup;
            for (i = 0; i < table.Cols.length; i++) xrefs[i] = -1;

            // Allocate a cursors for the main database table and for all indices. The index cursors might not be used, but if they are used they
            // need to occur right after the database cursor.  So go ahead and allocate enough space, just in case.
            int curId; // VDBE Cursor number of pTab
            tabList.Ids[0].Cursor = curId = parse.Tabs++;
            Index idx; // For looping over indices
            for (idx = table.Index; idx != null; idx = idx.Next)
                parse.Tabs++;

            // Initialize the name-context
            NameContext sNC = new NameContext(); // The name-context to resolve expressions in
            sNC.Parse = parse;
            sNC.SrcList = tabList;

            // Resolve the column names in all the expressions of the of the UPDATE statement.  Also find the column index
            // for each column to be updated in the pChanges array.  For each column to be updated, make sure we have authorization to change that column.
            bool chngRowid = false; // True if the record number is being changed
            Expr rowidExpr = null; // Expression defining the new record number
            for (i = 0; i < changes.Exprs; i++)
            {
                if (sqlite3ResolveExprNames(sNC, ref changes.Ids[i].Expr) != 0)
                    goto update_cleanup;
                for (j = 0; j < table.Cols.length; j++)
                {
                    if (string.Equals(table.Cols[j].Name, changes.Ids[i].Name, StringComparison.OrdinalIgnoreCase))
                    {
                        if (j == table.PKey)
                        {
                            chngRowid = true;
                            rowidExpr = changes.Ids[i].Expr;
                        }
                        xrefs[j] = i;
                        break;
                    }
                }
                if (j >= table.Cols.length)
                {
                    if (Expr::IsRowid(changes.Ids[i].Name))
                    {
                        chngRowid = true;
                        rowidExpr = changes.Ids[i].Expr;
                    }
                    else
                    {
                        parse.ErrorMsg("no such column: %s", changes.Ids[i].Name);
                        parse.CheckSchema = 1;
                        goto update_cleanup;
                    }
                }
#if !OMIT_AUTHORIZATION
                {
                    ARC rc = Auth.Check(parse, AUTH.UPDATE, table.Name, table.Cols[j].Name, ctx.DBs[db].Name);
                    if (rc == ARC.DENY) goto update_cleanup;
                    else if (rc == ARC.IGNORE) xrefs[j] = -1;
                }
#endif
            }

            bool hasFK = sqlite3FkRequired(parse, table, xrefs, chngRowid ? 1 : 0); // True if foreign key processing is required

            // Allocate memory for the array aRegIdx[].  There is one entry in the array for each index associated with table being updated.  Fill in
            // the value with a register number for indices that are to be used and with zero for unused indices.
            int idxLength;                   // Number of indices that need updating
            for (idxLength = 0, idx = table.Index; idx != null; idx = idx.Next, idxLength++) ;
            int[] regIdxs = null; // One register assigned to each index to be updated
            if (idxLength > 0)
            {
                regIdxs = new int[idxLength];
                if (regIdxs == null) goto update_cleanup;
            }
            for (j = 0, idx = table.Index; idx != null; idx = idx.Next, j++)
            {
                int regId;
                if (hasFK || chngRowid)
                    regId = ++parse.Mems;
                else
                {
                    regId = 0;
                    for (i = 0; i < idx.Columns.length; i++)
                        if (xrefs[idx.Columns[i]] >= 0)
                        {
                            regId = ++parse.Mems;
                            break;
                        }
                }
                regIdxs[j] = regId;
            }

            // Begin generating code.
            Vdbe v = parse.GetVdbe(); // The virtual database engine
            if (v == null) goto update_cleanup;
            if (parse.Nested == 0) v.CountChanges();
            parse.BeginWriteOperation(1, db);

#if !OMIT_VIRTUALTABLE
            // Virtual tables must be handled separately
            if (IsVirtual(table))
            {
                UpdateVirtualTable(parse, tabList, table, changes, rowidExpr, xrefs, where_, onError);
                where_ = null;
                tabList = null;
                goto update_cleanup;
            }
#endif

            // Register Allocations
            int regRowCount = 0;         // A count of rows changed
            int regOldRowid;             // The old rowid
            int regNewRowid;             // The new rowid
            int regNew;
            int regOld = 0;
            int regRowSet = 0;           // Rowset of rows to be updated

            // Allocate required registers.
            regOldRowid = regNewRowid = ++parse.Mems;
            if (trigger != null || hasFK)
            {
                regOld = parse.Mems + 1;
                parse.Mems += table.Cols.length;
            }
            if (chngRowid || trigger != null || hasFK)
                regNewRowid = ++parse.Mems;
            regNew = parse.Mems + 1;
            parse.Mems += table.Cols.length;

            // Start the view context.
            if (isView)
                Auth.ContextPush(parse, sContext, table.Name);

            // If we are trying to update a view, realize that view into a ephemeral table.
#if !OMIT_VIEW && !OMIT_TRIGGER
            if (isView)
                sqlite3MaterializeView(parse, table, where_, curId);
#endif

            // Resolve the column names in all the expressions in the WHERE clause.
            if (sqlite3ResolveExprNames(sNC, ref where_) != 0)
                goto update_cleanup;

            // Begin the database scan
            v.AddOp2(OP.Null, 0, regOldRowid);
            ExprList dummy = null;
            WhereInfo winfo = Where.Begin(parse, tabList, where_, ref dummy, WHERE.ONEPASS_DESIRED); // Information about the WHERE clause
            if (winfo == null) goto update_cleanup;
            bool okOnePass = winfo.OkOnePass; // True for one-pass algorithm without the FIFO

            // Remember the rowid of every item to be updated.
            v.AddOp2(OP.Rowid, curId, regOldRowid);
            if (!okOnePass)
                v.AddOp2(OP.RowSetAdd, regRowSet, regOldRowid);

            // End the database scan loop.
            Where.End(winfo);

            // Initialize the count of updated rows
            if ((ctx.Flags & Context.FLAG.CountRows) != 0 && parse.TriggerTab == null)
            {
                regRowCount = ++parse.Mems;
                v.AddOp2(OP.Integer, 0, regRowCount);
            }

            bool openAll = false; // True if all indices need to be opened
            if (!isView)
            {
                // Open every index that needs updating.  Note that if any index could potentially invoke a REPLACE conflict resolution 
                // action, then we need to open all indices because we might need to be deleting some records.
                if (!okOnePass) sqlite3OpenTable(parse, curId, db, table, OP.OpenWrite);
                if (onError == OE.Replace)
                    openAll = true;
                else
                {
                    openAll = false;
                    for (idx = table.Index; idx != null; idx = idx.Next)
                    {
                        if (idx.OnError == OE.Replace)
                        {
                            openAll = true;
                            break;
                        }
                    }
                }
                for (i = 0, idx = table.Index; idx != null; idx = idx.Next, i++)
                {
                    if (openAll || regIdxs[i] > 0)
                    {
                        KeyInfo key = sqlite3IndexKeyinfo(parse, idx);
                        v.AddOp4(OP.OpenWrite, curId + i + 1, idx.Id, db, key, Vdbe.P4T.KEYINFO_HANDOFF);
                        Debug.Assert(parse.Tabs > curId + i + 1);
                    }
                }
            }

            // Top of the update loop
            int addr = 0; // VDBE instruction address of the start of the loop
            if (okOnePass)
            {
                int a1 = v.AddOp1(OP.NotNull, regOldRowid);
                addr = v.AddOp0(OP.Goto);
                v.JumpHere(a1);
            }
            else
                addr = v.AddOp3(OP.RowSetRead, regRowSet, 0, regOldRowid);

            // Make cursor iCur point to the record that is being updated. If this record does not exist for some reason (deleted by a trigger,
            // for example, then jump to the next iteration of the RowSet loop.
            v.AddOp3(OP.NotExists, curId, addr, regOldRowid);

            // If the record number will change, set register regNewRowid to contain the new value. If the record number is not being modified,
            // then regNewRowid is the same register as regOldRowid, which is already populated.
            Debug.Assert(chngRowid || trigger != null || hasFK || regOldRowid == regNewRowid);
            if (chngRowid)
            {
                Expr.Code(parse, rowidExpr, regNewRowid);
                v.AddOp1(OP.MustBeInt, regNewRowid);
            }

            // If there are triggers on this table, populate an array of registers with the required old.* column data.
            if (hasFK || trigger != null)
            {
                uint oldmask = (hasFK ? sqlite3FkOldmask(parse, table) : 0);
                oldmask |= sqlite3TriggerColmask(parse, trigger, changes, 0, TRIGGER_BEFORE | TRIGGER_AFTER, table, onError);
                for (i = 0; i < table.Cols.length; i++)
                {
                    if (xrefs[i] < 0 || oldmask == 0xffffffff || (i < 32 && 0 != (oldmask & (1 << i))))
                        Expr.CodeGetColumnOfTable(v, table, curId, i, regOld + i);
                    else
                        v.AddOp2(OP.Null, 0, regOld + i);
                }
                if (!chngRowid)
                    v.AddOp2(OP.Copy, regOldRowid, regNewRowid);
            }

            // Populate the array of registers beginning at regNew with the new row data. This array is used to check constaints, create the new
            // table and index records, and as the values for any new.* references made by triggers.
            //
            // If there are one or more BEFORE triggers, then do not populate the registers associated with columns that are (a) not modified by
            // this UPDATE statement and (b) not accessed by new.* references. The values for registers not modified by the UPDATE must be reloaded from 
            // the database after the BEFORE triggers are fired anyway (as the trigger may have modified them). So not loading those that are not going to
            // be used eliminates some redundant opcodes.
            int newmask = (int)sqlite3TriggerColmask(parse, trigger, changes, 1, TRIGGER_BEFORE, table, onError); // Mask of NEW.* columns accessed by BEFORE triggers
            for (i = 0; i < table.Cols.length; i++)
            {
                if (i == table.PKey)
                {
                    //v.AddOp2(OP.Null, 0, regNew + i);
                }
                else
                {
                    j = xrefs[i];
                    if (j >= 0)
                        Expr.Code(parse, changes.Ids[j].Expr, regNew + i);
                    else if ((tmask & TRIGGER_BEFORE) == 0 || i > 31 || (newmask & (1 << i)) != 0)
                    {
                        // This branch loads the value of a column that will not be changed into a register. This is done if there are no BEFORE triggers, or
                        // if there are one or more BEFORE triggers that use this value via a new.* reference in a trigger program.
                        C.ASSERTCOVERAGE(i == 31);
                        C.ASSERTCOVERAGE(i == 32);
                        v.AddOp3(OP.Column, curId, i, regNew + i);
                        v.ColumnDefault(table, i, regNew + i);
                    }
                }
            }

            // Fire any BEFORE UPDATE triggers. This happens before constraints are verified. One could argue that this is wrong.
            if ((tmask & TRIGGER_BEFORE) != 0)
            {
                v.AddOp2(OP.Affinity, regNew, table.Cols.length);
                sqlite3TableAffinityStr(v, table);
                sqlite3CodeRowTrigger(parse, trigger, TK.UPDATE, changes, TRIGGER_BEFORE, table, regOldRowid, onError, addr);

                // The row-trigger may have deleted the row being updated. In this case, jump to the next row. No updates or AFTER triggers are 
                // required. This behavior - what happens when the row being updated is deleted or renamed by a BEFORE trigger - is left undefined in the documentation.
                v.AddOp3(OP.NotExists, curId, addr, regOldRowid);

                // If it did not delete it, the row-trigger may still have modified some of the columns of the row being updated. Load the values for 
                // all columns not modified by the update statement into their registers in case this has happened.
                for (i = 0; i < table.Cols.length; i++)
                    if (xrefs[i] < 0 && i != table.PKey)
                    {
                        v.AddOp3(OP.Column, curId, i, regNew + i);
                        v.ColumnDefault(table, i, regNew + i);
                    }
            }

            if (!isView)
            {
                // Do constraint checks.
                int dummy2;
                sqlite3GenerateConstraintChecks(parse, table, curId, regNewRowid, regIdxs, (chngRowid ? regOldRowid : 0), true, onError, addr, out dummy2);

                // Do FK constraint checks.
                if (hasFK)
                    sqlite3FkCheck(parse, table, regOldRowid, 0);

                // Delete the index entries associated with the current record.
                int j1 = v.AddOp3(OP.NotExists, curId, 0, regOldRowid); // Address of jump instruction
                sqlite3GenerateRowIndexDelete(parse, table, curId, regIdxs);

                // If changing the record number, delete the old record.
                if (hasFK || chngRowid)
                    v.AddOp2(OP.Delete, curId, 0);
                v.JumpHere(j1);

                if (hasFK)
                    sqlite3FkCheck(parse, table, 0, regNewRowid);

                // Insert the new index entries and the new record.
                sqlite3CompleteInsertion(parse, table, curId, regNewRowid, regIdxs, true, false, false);

                // Do any ON CASCADE, SET NULL or SET DEFAULT operations required to handle rows (possibly in other tables) that refer via a foreign key
                // to the row just updated.
                if (hasFK)
                    sqlite3FkActions(parse, table, changes, regOldRowid);
            }

            // Increment the row counter 
            if ((ctx.Flags & Context.FLAG.CountRows) != 0 && parse.TriggerTab == null)
                v.AddOp2(OP.AddImm, regRowCount, 1);

            sqlite3CodeRowTrigger(parse, trigger, TK.UPDATE, changes, TRIGGER_AFTER, table, regOldRowid, onError, addr);

            // Repeat the above with the next record to be updated, until all record selected by the WHERE clause have been updated.
            v.AddOp2(OP.Goto, 0, addr);
            v.JumpHere(addr);

            // Close all tables
            Debug.Assert(regIdxs != null);
            for (i = 0, idx = table.Index; idx != null; idx = idx.Next, i++)
                if (openAll || regIdxs[i] > 0)
                    v.AddOp2(OP.Close, curId + i + 1, 0);
            v.AddOp2(OP.Close, curId, 0);

            // Update the sqlite_sequence table by storing the content of the maximum rowid counter values recorded while inserting into
            // autoincrement tables.
            if (parse.Nested == 0 && parse.TriggerTab == null)
                sqlite3AutoincrementEnd(parse);

            // Return the number of rows that were changed. If this routine is generating code because of a call to sqlite3NestedParse(), do not
            // invoke the callback function.
            if ((ctx.Flags & Context.FLAG.CountRows) != 0 && parse.TriggerTab == null && parse.Nested == 0)
            {
                v.AddOp2(OP.ResultRow, regRowCount, 1);
                v.SetNumCols(1);
                v.SetColName(0, COLNAME_NAME, "rows updated", SQLITE_STATIC);
            }

        update_cleanup:
#if !OMIT_AUTHORIZATION
            Auth.ContextPop(sContext);
#endif
            C._tagfree(ctx, ref regIdxs);
            C._tagfree(ctx, ref xrefs);
            SrcList.Delete(ctx, ref tabList);
            ExprList.Delete(ctx, ref changes);
            Expr.Delete(ctx, ref where_);
            return;
        }

#if !OMIT_VIRTUALTABLE

        static void UpdateVirtualTable(Parse parse, SrcList src, Table table, ExprList changes, Expr rowid, int[] xrefs, Expr where_, int onError)
        {
            int i;
            Context ctx = parse.Ctx; // Database connection
            VTable vtable = VTable.GetVTable(ctx, table);
            SelectDest dest = new SelectDest();

            // Construct the SELECT statement that will find the new values for all updated rows.
            ExprList list = ExprList.Append(parse, 0, Expr.Expr(ctx, TK.ID, "_rowid_")); // The result set of the SELECT statement
            if (rowid != null)
                list = ExprList.Append(parse, list, Expr.Dup(ctx, rowid, 0));
            Debug.Assert(table.PKey < 0);
            for (i = 0; i < table.Cols.length; i++)
            {
                Expr expr = (xrefs[i] >= 0 ? Expr.Dup(ctx, changes.Ids[xrefs[i]].Expr, 0) : Expr.Expr(ctx, TK.ID, table.Cols[i].Name)); // Temporary expression
                list = ExprList.Append(parse, list, expr);
            }
            Select select = Select.New(parse, list, src, where_, null, null, null, 0, null, null); // The SELECT statement

            // Create the ephemeral table into which the update results will be stored.
            Vdbe v = parse.V; // Virtual machine under construction
            Debug.Assert(v != null);
            int ephemTab = parse.Tabs++; // Table holding the result of the SELECT
            v.AddOp2(OP.OpenEphemeral, ephemTab, table.Cols.length + 1 + (rowid != null ? 1 : 0));
            v.ChangeP5(BTREE_UNORDERED);

            // fill the ephemeral table 
            Select.DestInit(dest, SRT.Table, ephemTab);
            Select.Select(parse, select, ref dest);

            // Generate code to scan the ephemeral table and call VUpdate.
            int regId = ++parse.Mems;// First register in set passed to OP_VUpdate
            parse.Mems += table.Cols.length + 1;
            int addr = v.AddOp2(OP.Rewind, ephemTab, 0); // Address of top of loop
            v.AddOp3(OP.Column, ephemTab, 0, regId);
            v.AddOp3(OP.Column, ephemTab, (rowid != null ? 1 : 0), regId + 1);
            for (i = 0; i < table.nCol; i++)
                v.AddOp3(OP.Column, ephemTab, i + 1 + (rowid != null ? 1 : 0), regId + 2 + i);
            sqlite3VtabMakeWritable(parse, table);
            v.AddOp4(OP_VUpdate, 0, table.Cols.length + 2, regId, vtable, P4_VTAB);
            v.ChangeP5((byte)(onError == OE_Default ? OE_Abort : onError));
            parse.MayAbort();
            v.AddOp2(OP.Next, ephemTab, addr + 1);
            v.JumpHere(addr);
            v.AddOp2(OP.Close, ephemTab, 0);

            // Cleanup
            Select.Delete(ctx, ref select);
        }
#endif
    }
}
