using System;
using System.Diagnostics;
using System.Text;

namespace Core.Command
{
    public partial class Delete
    {
        public static Table SrcListLookup(Parse parse, SrcList src)
        {
            SrcList.SrcListItem item = src.Ids[0];
            Debug.Assert(item != null && src.Srcs == 1);
            Table table = parse.LocateTableItem(false, item);
            Parse.DeleteTable(parse.Ctx, ref item.Table);
            item.Table = table;
            if (table != null)
                table.Refs++;
            if (Select.IndexedByLookup(parse, item) != 0)
                table = null;
            return table;
        }

        public static bool IsReadOnly(Parse parse, Table table, bool viewOk)
        {
            // A table is not writable under the following circumstances:
            //   1) It is a virtual table and no implementation of the xUpdate method has been provided, or
            //   2) It is a system table (i.e. sqlite_master), this call is not part of a nested parse and writable_schema pragma has not 
            //      been specified.
            // In either case leave an error message in pParse and return non-zero.
            if ((E.IsVirtual(table) && VTable.GetVTable(parse.Ctx, table).Module.IModule.Update == null) ||
               ((table.TabFlags & TF.Readonly) != 0 && (parse.Ctx.Flags & Context.FLAG.WriteSchema) == 0 && parse.Nested == 0))
            {
                parse.ErrorMsg("table %s may not be modified", table.Name);
                return true;
            }

#if !OMIT_VIEW
            if (!viewOk && table.Select != null)
            {
                parse.ErrorMsg("cannot modify %s because it is a view", table.Name);
                return true;
            }
#endif
            return false;
        }

#if !OMIT_VIEW && !OMIT_TRIGGER
        public static void MaterializeView(Parse parse, Table view, Expr where_, int curId)
        {
            Context ctx = parse.Ctx;
            int db = Prepare.SchemaToIndex(ctx, view.Schema);

            where_ = Expr.Dup(ctx, where_, 0);
            SrcList from = Parse.SrcListAppend(ctx, null, null, null);

            if (from != null)
            {
                Debug.Assert(from.Srcs == 1);
                from.Ids[0].Name = view.Name;
                from.Ids[0].Database = ctx.DBs[db].Name;
                Debug.Assert(from.Ids[0].On == null);
                Debug.Assert(from.Ids[0].Using == null);
            }

            Select select = Select.New(parse, 0, from, where_, 0, 0, 0, 0, 0, 0);
            if (select != null) select.SelFlags |= SF.Materialize;

            SelectDest dest = new SelectDest();
            Select.DestInit(dest, SRT.EphemTab, curId);
            Select.Select(parse, select, dest);
            Select.Delete(ctx, select);
        }
#endif

#if true || ENABLE_UPDATE_DELETE_LIMIT && !OMIT_SUBQUERY
        public Expr LimitWhere(Parse parse, SrcList src, Expr where_, ExprList orderBy, Expr limit, Expr offset, char stmtType)
        {
            // Check that there isn't an ORDER BY without a LIMIT clause.
            if (orderBy != null && (limit == null))
            {
                parse.ErrorMsg("ORDER BY without LIMIT on %s", stmtType);
                goto limit_where_cleanup_2;
            }

            // We only need to generate a select expression if there is a limit/offset term to enforce.
            if (limit == null)
            {
                Debug.Assert(offset == null); // if pLimit is null, pOffset will always be null as well.
                return where_;
            }

            // Generate a select expression tree to enforce the limit/offset term for the DELETE or UPDATE statement.  For example:
            //   DELETE FROM table_a WHERE col1=1 ORDER BY col2 LIMIT 1 OFFSET 1
            // becomes:
            //   DELETE FROM table_a WHERE rowid IN ( 
            //     SELECT rowid FROM table_a WHERE col1=1 ORDER BY col2 LIMIT 1 OFFSET 1
            //   );
            Expr selectRowid = Expr.PExpr_(parse, TK.ROW, null, null, null); // SELECT rowid ...
            if (selectRowid == null) goto limit_where_cleanup_2;
            ExprList elist = ExprList.Append(parse, null, selectRowid); // Expression list contaning only pSelectRowid
            if (elist == null) goto limit_where_cleanup_2;

            // duplicate the FROM clause as it is needed by both the DELETE/UPDATE tree and the SELECT subtree.
            SrcList selectSrc = SrcList.Dup(parse.Ctx, src, 0); // SELECT rowid FROM x ... (dup of pSrc)
            if (selectSrc == null)
            {
                ExprList.Delete(parse.Ctx, elist);
                goto limit_where_cleanup_2;
            }

            // generate the SELECT expression tree.
            Select select = Select.New(parse, elist, selectSrc, where_, null, null, orderBy, 0, limit, offset); // Complete SELECT tree
            if (select == null) return null;

            // now generate the new WHERE rowid IN clause for the DELETE/UDPATE
            Expr whereRowid = Expr.PExpr_(parse, TK.ROW, null, null, null); // WHERE rowid ..
            if (whereRowid == null) goto limit_where_cleanup_1;
            Expr inClause = Expr.PExpr_(parse, TK.IN, whereRowid, null, null); // WHERE rowid IN ( select )
            if (inClause == null) goto limit_where_cleanup_1;

            inClause.x.Select = select;
            inClause.Flags |= EP.xIsSelect;
            Expr.SetHeight(parse, inClause);
            return inClause;

        // something went wrong. clean up anything allocated.
        limit_where_cleanup_1:
            Select.Delete(parse.Ctx, select);
            return null;

        limit_where_cleanup_2:
            Expr.Delete(parse.Ctx, ref where_);
            ExprList.Delete(parse.Ctx, orderBy);
            Expr.Delete(parse.Ctx, ref limit);
            Expr.Delete(parse.Ctx, ref offset);
            return null;
        }
#endif

        public static void DeleteFrom(Parse parse, SrcList tabList, Expr where_)
        {
            AuthContext sContext = new AuthContext(); // Authorization context
            Context ctx = parse.Ctx; // Main database structure
            if (parse.Errs != 0 || ctx.MallocFailed)
                goto delete_from_cleanup;
            Debug.Assert(tabList.Srcs == 1);

            // Locate the table which we want to delete.  This table has to be put in an SrcList structure because some of the subroutines we
            // will be calling are designed to work with multiple tables and expect an SrcList* parameter instead of just a Table* parameter.
            Table table = SrcList.Lookup(parse, tabList); // The table from which records will be deleted
            if (table == null) goto delete_from_cleanup;

            // Figure out if we have any triggers and if the table being deleted from is a view
#if !OMIT_TRIGGER
            int dummy;
            Trigger trigger = Triggers.Exist(parse, table, TK.DELETE, null, out dummy); // List of table triggers, if required
#if OMIT_VIEW
            const bool isView = false;
#else
            bool isView = (table.Select != null); // True if attempting to delete from a view
#endif
#else
            const Trigger trigger = null;
            bool isView = false;
#endif

            // If pTab is really a view, make sure it has been initialized.
            if (sqlite3ViewGetColumnNames(parse, table) != null || IsReadOnly(parse, table, (trigger != null)))
                goto delete_from_cleanup;
            int db = sqlite3SchemaToIndex(ctx, table.Schema); // Database number
            Debug.Assert(db < ctx.DBs.length);
            string dbName = ctx.DBs[db].Name; // Name of database holding pTab
            ARC rcauth = Auth.Check(parse, AUTH.DELETE, table.Name, 0, dbName); // Value returned by authorization callback
            Debug.Assert(rcauth == ARC.OK || rcauth == ARC.DENY || rcauth == ARC.IGNORE);
            if (rcauth == ARC.DENY)
                goto delete_from_cleanup;
            Debug.Assert(!isView || trigger != null);

            // Assign cursor number to the table and all its indices.
            Debug.Assert(tabList.Srcs == 1);
            int curId = tabList.Ids[0].Cursor = parse.Tabs++; // VDBE VdbeCursor number for pTab
            Index idx; // For looping over indices of the table
            for (idx = table.Index; idx != null; idx = idx.Next)
                parse.Tabs++;

            // Start the view context
            if (isView)
                Auth.ContextPush(parse, sContext, table.Name);

            // Begin generating code.
            Vdbe v = parse.GetVdbe(); // The virtual database engine 
            if (v == null)
                goto delete_from_cleanup;
            if (parse.Nested == 0) v.CountChanges();
            parse.BeginWriteOperation(1, db);

            // If we are trying to delete from a view, realize that view into a ephemeral table.
#if !OMIT_VIEW && !OMIT_TRIGGER
            if (isView)
                MaterializeView(parse, table, where_, curId);
#endif
            // Resolve the column names in the WHERE clause.
            NameContext sNC = new NameContext(); // Name context to resolve expressions in
            sNC.Parse = parse;
            sNC.SrcList = tabList;
            if (sqlite3ResolveExprNames(sNC, ref where_) != 0)
                goto delete_from_cleanup;

            // Initialize the counter of the number of rows deleted, if we are counting rows.
            int memCnt = -1; // Memory cell used for change counting
            if ((ctx.Flags & Context.FLAG.CountRows) != 0)
            {
                memCnt = ++parse.Mems;
                v.AddOp2(OP.Integer, 0, memCnt);
            }

#if !OMIT_TRUNCATE_OPTIMIZATION
            // Special case: A DELETE without a WHERE clause deletes everything. It is easier just to erase the whole table. Prior to version 3.6.5,
            // this optimization caused the row change count (the value returned by API function sqlite3_count_changes) to be set incorrectly.
            if (rcauth == ARC.OK && where_ == null && trigger == null && !IsVirtual(table) && !FKey.FkRequired(parse, table, null, 0))
            {
                Debug.Assert(!isView);
                v.AddOp4(OP.Clear, table.Id, db, memCnt, table.Name, Vdbe.P4T.STATIC);
                for (idx = table.Index; idx != null; idx = idx.Next)
                {
                    Debug.Assert(idx.Schema == table.Schema);
                    v.AddOp2(OP.Clear, idx.Id, db);
                }
            }
            else
#endif
            // The usual case: There is a WHERE clause so we have to scan through the table and pick which records to delete.
            {
                int rowSet = ++parse.Mems; // Register for rowset of rows to delete
                int rowid = ++parse.Mems; // Used for storing rowid values.

                // Collect rowids of every row to be deleted.
                v.AddOp2(OP.Null, 0, rowSet);
                ExprList dummy = null;
                WhereInfo winfo = Where.Begin(parse, tabList, where_, ref dummy, WHERE_DUPLICATES_OK, 0); // Information about the WHERE clause
                if (winfo == null) goto delete_from_cleanup;
                int regRowid = Expr.CodeGetColumn(parse, table, -1, curId, rowid); // Actual register containing rowids
                v.AddOp2(OP.RowSetAdd, rowSet, regRowid);
                if ((ctx.Flags & Context.FLAG.CountRows) != 0)
                    v.AddOp2(OP.AddImm, memCnt, 1);
                Where.End(winfo);

                // Delete every item whose key was written to the list during the database scan.  We have to delete items after the scan is complete
                // because deleting an item can change the scan order.
                int end = v.MakeLabel();

                // Unless this is a view, open cursors for the table we are deleting from and all its indices. If this is a view, then the
                // only effect this statement has is to fire the INSTEAD OF triggers.
                if (!isView)
                    sqlite3OpenTableAndIndices(parse, table, curId, OP.OpenWrite);
                int addr = v.AddOp3(OP.RowSetRead, rowSet, end, rowid);

                // Delete the row
#if !OMIT_VIRTUALTABLE
                if (IsVirtual(table))
                {
                    VTable vtable = VTable.GetVTable(ctx, table);
                    VTable.MakeWritable(parse, table);
                    v.AddOp4(OP.VUpdate, 0, 1, rowid, vtable, Vdbe.P4T.VTAB);
                    v.ChangeP5(OE.Abort);
                    sqlite3MayAbort(parse);
                }
                else
#endif
                {
                    int count = (parse.Nested == 0; // True to count changes
                    GenerateRowDelete(parse, table, curId, rowid, count, trigger, OE.Default);
                }

                // End of the delete loop
                v.AddOp2(OP.Goto, 0, addr);
                v.ResolveLabel(end);

                // Close the cursors open on the table and its indexes.
                if (!isView && !IsVirtual(table))
                {
                    for (int i = 1, idx = table.Index; idx != null; i++, idx = idx.Next)
                        v.AddOp2(OP.Close, curId + i, idx.Id);
                    v.AddOp1(OP.Close, curId);
                }
            }

            // Update the sqlite_sequence table by storing the content of the maximum rowid counter values recorded while inserting into
		    // autoincrement tables.
            if (parse.Nested == 0 && parse.TriggerTab == null)
                sqlite3AutoincrementEnd(parse);

            // Return the number of rows that were deleted. If this routine is generating code because of a call to sqlite3NestedParse(), do not
		    // invoke the callback function.
            if ((ctx.Flags & Context.FLAG.CountRows) != 0 && parse.Nested == 0 && parse.TriggerTab == null)
            {
                v.AddOp2(OP.ResultRow, memCnt, 1);
                v.SetNumCols(1);
                v.SetColName(0, COLNAME_NAME, "rows deleted", SQLITE_STATIC);
            }

        delete_from_cleanup:
            Auth.ContextPop(sContext);
            SrcList.Delete(ctx, ref tabList);
            Expr.Delete(ctx, ref where_);
            return;
        }

        public static void GenerateRowDelete(Parse parse, Table table, int curId, int rowid, int count, Trigger trigger, OE onconf)
        {
            // Vdbe is guaranteed to have been allocated by this stage.
            Vdbe v = parse.V;
            Debug.Assert(v != null);

            // Seek cursor iCur to the row to delete. If this row no longer exists (this can happen if a trigger program has already deleted it), do
            // not attempt to delete it or fire any DELETE triggers.
            int label = v.MakeLabel(); // Label resolved to end of generated code
            v.AddOp3(OP.NotExists, curId, label, rowid);

            // If there are any triggers to fire, allocate a range of registers to use for the old.* references in the triggers.
            if (FKey.FkRequired(parse, table, null, 0) != 0 || trigger != null)
            {
                // TODO: Could use temporary registers here. Also could attempt to avoid copying the contents of the rowid register.
                uint mask = sqlite3TriggerColmask(parse, trigger, null, 0, TRIGGER.BEFORE | TRIGGER.AFTER, table, onconf); // Mask of OLD.* columns in use
                mask |= sqlite3FkOldmask(parse, table);
                int oldId = parse.Mems + 1; // First register in OLD.* array
                parse.Mems += (1 + table.Cols.length);

                // Populate the OLD.* pseudo-table register array. These values will be used by any BEFORE and AFTER triggers that exist.
                v.AddOp2(OP.Copy, rowid, oldId);
                for (int col = 0; col < table.Cols.length; col++) // Iterator used while populating OLD.*
                    if (mask == 0xffffffff || (mask & (1 << col)) != 0)
                        Expr.CodeGetColumnOfTable(v, table, curId, col, oldId + col + 1);

                // Invoke BEFORE DELETE trigger programs.
                sqlite3CodeRowTrigger(parse, trigger, TK.DELETE, null, TRIGGER.BEFORE, table, oldId, onconf, label);

                // Seek the cursor to the row to be deleted again. It may be that the BEFORE triggers coded above have already removed the row
                // being deleted. Do not attempt to delete the row a second time, and do not fire AFTER triggers.
                v.AddOp3(OP.NotExists, curId, label, rowid);

                // Do FK processing. This call checks that any FK constraints that refer to this table (i.e. constraints attached to other tables) are not violated by deleting this row.
                FKey.FkCheck(parse, table, oldId, 0);
            }

            // Delete the index and table entries. Skip this step if table is really a view (in which case the only effect of the DELETE statement is to fire the INSTEAD OF triggers).
            if (table.Select == null)
            {
                GenerateRowIndexDelete(parse, table, curId, null);
                v.AddOp2(OP.Delete, curId, (count != 0 ? (int)OPFLAG.NCHANGE : 0));
                if (count != 0)
                    v.ChangeP4(-1, table.Name, Vdbe.P4T.TRANSIENT);
            }

            // Do any ON CASCADE, SET NULL or SET DEFAULT operations required to handle rows (possibly in other tables) that refer via a foreign key to the row just deleted.
            FKey.FkActions(parse, table, null, oldId);

            // Invoke AFTER DELETE trigger programs.
            sqlite3CodeRowTrigger(parse, trigger, TK.DELETE, null, TRIGGER.AFTER, table, oldId, onconf, label);

            // Jump here if the row had already been deleted before any BEFORE trigger programs were invoked. Or if a trigger program throws a RAISE(IGNORE) exception.
            v.ResolveLabel(label);
        }

        //public static void GenerateRowIndexDelete(Parse parse, Table table, int curId, int nothing) { int[] regIdxs = null; GenerateRowIndexDelete(parse, table, curId, regIdxs); }
        public static void GenerateRowIndexDelete(Parse parse, Table table, int curId, int[] regIdxs)
        {
            int i;
            Index idx;
            for (i = 1, idx = table.Index; idx != null; i++, idx = idx.Next)
            {
                if (regIdxs != null && regIdxs[i - 1] == 0)
                    continue;
                int r1 = GenerateIndexKey(parse, idx, curId, 0, false);
                parse.V.AddOp3(OP.IdxDelete, curId + i, r1, idx.Columns.length + 1);
            }
        }

        public static int GenerateIndexKey(Parse parse, Index index, int curId, int regOut, bool doMakeRec)
        {
            Vdbe v = parse.V;
            Table table = index.Table;

            int cols = index.Columns.length;
            int regBase = Expr.GetTempRange(parse, cols + 1);
            v.AddOp2(OP.Rowid, curId, regBase + cols);
            for (int j = 0; j < cols; j++)
            {
                int idx = index.Columns[j];
                if (idx == table.PKey)
                    v.AddOp2(OP.SCopy, regBase + cols, regBase + j);
                else
                {
                    v.AddOp3(OP.Column, curId, idx, regBase + j);
                    v.ColumnDefault(table, idx, -1);
                }
            }
            if (doMakeRec)
            {
                string affName = (table.Select != null || E.CtxOptimizationDisabled(parse.Ctx, SQLITE.IdxRealAsInt) ? null : sqlite3IndexAffinityStr(v, index));
                v.AddOp3(OP.MakeRecord, regBase, cols + 1, regOut);
                v.ChangeP4(-1, affName, Vdbe.P4T.TRANSIENT);
            }
            Expr.ReleaseTempRange(parse, regBase, cols + 1);
            return regBase;
        }
    }
}