#if !OMIT_FOREIGN_KEY
using System;
using System.Diagnostics;
using System.Text;
using Bitmask = System.UInt64;

namespace Core
{
    public partial class Parse
    {
#if !OMIT_TRIGGER
        int LocateFkeyIndex(Table parent, FKey fkey, out Index indexOut, out int[] colsOut)
        {
            indexOut = null; colsOut = null;

            int colsLength = fkey.Cols.length; // Number of columns in parent key
            string key = fkey.Cols[0].Col; // Name of left-most parent key column

            // The caller is responsible for zeroing output parameters.
            //: _assert(indexOut && *indexOut == nullptr);
            //: _assert(!colsOut || *colsOut == nullptr);

            // If this is a non-composite (single column) foreign key, check if it maps to the INTEGER PRIMARY KEY of table pParent. If so, leave *ppIdx 
            // and *paiCol set to zero and return early. 
            //
            // Otherwise, for a composite foreign key (more than one column), allocate space for the aiCol array (returned via output parameter *paiCol).
            // Non-composite foreign keys do not require the aiCol array.
            int[] cols = null; // Value to return via *paiCol
            if (colsLength == 1)
            {
                // The FK maps to the IPK if any of the following are true:
                //   1) There is an INTEGER PRIMARY KEY column and the FK is implicitly mapped to the primary key of table pParent, or
                //   2) The FK is explicitly mapped to a column declared as INTEGER PRIMARY KEY.
                if (parent.PKey >= 0)
                    if (key == null || string.Equals(parent.Cols[parent.PKey].Name, key, StringComparison.InvariantCultureIgnoreCase)) return 0;
            }
            else //: if (colsOut)
            {
                Debug.Assert(colsLength > 1);
                cols = new int[colsLength]; //: (int *)_tagalloc(Ctx, colsLength*sizeof(int));
                if (cols == null) return 1;
                colsOut = cols;
            }

            Index index = null; // Value to return via *ppIdx
            for (index = parent.Index; index != null; index = index.Next)
            {
                if (index.Columns.length == colsLength && index.OnError != OE.None)
                {
                    // pIdx is a UNIQUE index (or a PRIMARY KEY) and has the right number of columns. If each indexed column corresponds to a foreign key
                    // column of pFKey, then this index is a winner.
                    if (key == null)
                    {
                        // If zKey is NULL, then this foreign key is implicitly mapped to the PRIMARY KEY of table pParent. The PRIMARY KEY index may be 
                        // identified by the test (Index.autoIndex==2).
                        if (index.AutoIndex == 2)
                        {
                            if (cols != null)
                                for (int i = 0; i < colsLength; i++) cols[i] = fkey.Cols[i].From;
                            break;
                        }
                    }
                    else
                    {
                        // If zKey is non-NULL, then this foreign key was declared to map to an explicit list of columns in table pParent. Check if this
                        // index matches those columns. Also, check that the index uses the default collation sequences for each column.
                        int i, j;
                        for (i = 0; i < colsLength; i++)
                        {
                            int col = index.Columns[i]; // Index of column in parent tbl
                            // If the index uses a collation sequence that is different from the default collation sequence for the column, this index is
                            // unusable. Bail out early in this case.
                            string dfltColl = parent.Cols[col].Coll; // Def. collation for column
                            if (string.IsNullOrEmpty(dfltColl))
                                dfltColl = "BINARY";
                            if (!string.Equals(index.CollNames[i], dfltColl, StringComparison.InvariantCultureIgnoreCase)) break;

                            string indexCol = parent.Cols[col].Name; // Name of indexed column
                            for (j = 0; j < colsLength; j++)
                            {
                                if (string.Equals(fkey.Cols[j].Col, indexCol, StringComparison.InvariantCultureIgnoreCase))
                                {
                                    if (cols != null) cols[i] = fkey.Cols[j].From;
                                    break;
                                }
                            }
                            if (j == colsLength) break;
                        }
                        if (i == colsLength) break; // pIdx is usable
                    }
                }
            }

            if (index == null)
            {
                if (!DisableTriggers)
                    ErrorMsg("foreign key mismatch - \"%w\" referencing \"%w\"", fkey.From.Name, fkey.To);
                C._tagfree(Ctx, ref cols);
                return 1;
            }

            indexOut = index;
            return 0;
        }

        static void FKLookupParent(Parse parse, int db, Table table, Index index, FKey fkey, int[] cols, int regDataId, int incr, bool isIgnore)
        {
            Vdbe v = parse.GetVdbe(); // Vdbe to add code to
            int curId = parse.Tabs - 1; // Cursor number to use
            int okId = v.MakeLabel(); // jump here if parent key found

            // If nIncr is less than zero, then check at runtime if there are any outstanding constraints to resolve. If there are not, there is no need
            // to check if deleting this row resolves any outstanding violations.
            //
            // Check if any of the key columns in the child table row are NULL. If any are, then the constraint is considered satisfied. No need to 
            // search for a matching row in the parent table.
            int i;
            if (incr < 0)
                v.AddOp2(OP.FkIfZero, fkey.IsDeferred, okId);
            for (i = 0; i < fkey.Cols.length; i++)
            {
                int regId = cols[i] + regDataId + 1;
                v.AddOp2(OP.IsNull, regId, okId);
            }

            if (!isIgnore)
            {
                if (index == null)
                {
                    // If pIdx is NULL, then the parent key is the INTEGER PRIMARY KEY column of the parent table (table pTab).
                    int mustBeIntId; // Address of MustBeInt instruction
                    int regTempId = parse.GetTempReg();

                    // Invoke MustBeInt to coerce the child key value to an integer (i.e. apply the affinity of the parent key). If this fails, then there
                    // is no matching parent key. Before using MustBeInt, make a copy of the value. Otherwise, the value inserted into the child key column
                    // will have INTEGER affinity applied to it, which may not be correct.
                    v.AddOp2(OP.SCopy, cols[0] + 1 + regDataId, regTempId);
                    mustBeIntId = v.AddOp2(OP.MustBeInt, regTempId, 0);

                    // If the parent table is the same as the child table, and we are about to increment the constraint-counter (i.e. this is an INSERT operation),
                    // then check if the row being inserted matches itself. If so, do not increment the constraint-counter.
                    if (table == fkey.From && incr == 1)
                        v.AddOp3(OP.Eq, regDataId, okId, regTempId);

                    parse.OpenTable(parse, curId, db, table, OP.OpenRead);
                    v.AddOp3(OP.NotExists, curId, 0, regTempId);
                    v.AddOp2(OP.Goto, 0, okId);
                    v.JumpHere(v.CurrentAddr() - 2);
                    v.JumpHere(mustBeIntId);
                    parse.ReleaseTempReg(regTempId);
                }
                else
                {
                    int colsLength = fkey.Cols.length;
                    int regTempId = parse.GetTempRange(colsLength);
                    int regRecId = parse.GetTempReg();
                    KeyInfo key = IndexKeyinfo(parse, index);

                    v.AddOp3(OP.OpenRead, curId, index.Id, db);
                    v.ChangeP4(v, -1, key, Vdbe.P4T.KEYINFO_HANDOFF);
                    for (i = 0; i < colsLength; i++)
                        v.AddOp2(OP.Copy, cols[i] + 1 + regDataId, regTempId + i);

                    // If the parent table is the same as the child table, and we are about to increment the constraint-counter (i.e. this is an INSERT operation),
                    // then check if the row being inserted matches itself. If so, do not increment the constraint-counter. 

                    // If any of the parent-key values are NULL, then the row cannot match itself. So set JUMPIFNULL to make sure we do the OP_Found if any
                    // of the parent-key values are NULL (at this point it is known that none of the child key values are).
                    if (table == fkey.From && incr == 1)
                    {
                        int jumpId = v.CurrentAddr() + colsLength + 1;
                        for (i = 0; i < colsLength; i++)
                        {
                            int childId = cols[i] + 1 + regDataId;
                            int parentId = index.Columns[i] + 1 + regDataId;
                            Debug.Assert(cols[i] != table.PKey);
                            if (index.Columns[i] == table.PKey)
                                parentId = regDataId;  // The parent key is a composite key that includes the IPK column
                            v.AddOp3(OP.Ne, childId, jumpId, parentId);
                            v.ChangeP5(SQLITE_JUMPIFNULL);
                        }
                        v.AddOp2(OP.Goto, 0, okId);
                    }

                    v.AddOp3(OP.MakeRecord, regTempId, colsLength, regRecId);
                    v.ChangeP4(-1, IndexAffinityStr(v, index), Vdbe.P4T.TRANSIENT);
                    v.AddOp4Int(OP.Found, curId, okId, regRecId, 0);

                    parse.ReleaseTempReg(regRecId);
                    parse.ReleaseTempRange(regTempId, colsLength);
                }
            }

            if (!fkey.IsDeferred && parse.Toplevel == null && parse.IsMultiWrite == 0)
            {
                // Special case: If this is an INSERT statement that will insert exactly one row into the table, raise a constraint immediately instead of
                // incrementing a counter. This is necessary as the VM code is being generated for will not open a statement transaction.
                Debug.Assert(incr == 1);
                HaltConstraint(parse, OE.Abort, "foreign key constraint failed", Vdbe.P4T.STATIC);
            }
            else
            {
                if (incr > 0 && !fkey.IsDeferred)
                    E.Parse_Toplevel(parse).MayAbort = true;
                v.AddOp2(OP.FkCounter, fkey.IsDeferred, incr);
            }

            v.ResolveLabel(v, okId);
            v.AddOp1(OP.Close, curId);
        }

        static void FKScanChildren(Parse parse, SrcList src, Table table, Index index, FKey fkey, int[] cols, int regDataId, int incr)
        {
            Context ctx = parse.Ctx; // Database handle
            Vdbe v = parse.GetVdbe();
            Expr where_ = null; // WHERE clause to scan with

            Debug.Assert(index == null || index.Table == table);
            int fkIfZero = 0; // Address of OP_FkIfZero
            if (incr < 0)
                fkIfZero = v.AddOp2(OP.FkIfZero, fkey.IsDeferred, 0);

            // Create an Expr object representing an SQL expression like:
            //
            //   <parent-key1> = <child-key1> AND <parent-key2> = <child-key2> ...
            //
            // The collation sequence used for the comparison should be that of the parent key columns. The affinity of the parent key column should
            // be applied to each child key value before the comparison takes place.
            for (int i = 0; i < fkey.Cols.length; i++)
            {
                int col; // Index of column in child table
                Expr left = Expr.Expr(ctx, TK.REGISTER, null); // Value from parent table row
                if (left != null)
                {
                    // Set the collation sequence and affinity of the LHS of each TK_EQ expression to the parent key column defaults.
                    if (index != null)
                    {
                        col = index.Columns[i];
                        Column colObj = table.Cols[col];
                        if (table.PKey == col) col = -1;
                        left.TableId = regDataId + col + 1;
                        left.Aff = colObj.Affinity;
                        string collName = colObj.Coll;
                        if (collName == null) collName = ctx.DefaultColl.Name;
                        left = Expr.AddCollateString(parse, left, collName);
                    }
                    else
                    {
                        left.TableId = regDataId;
                        left.Aff = AFF.INTEGER;
                    }
                }
                col = (cols != null ? cols[i] : fkey.Cols[0].From);
                Debug.Assert(col >= 0);
                string colName = fkey.From.Cols[col].Name; // Name of column in child table
                Expr right = Expr.Expr(ctx, TK.ID, colName); // Column ref to child table
                Expr eq = Expr.PExpr(parse, TK.EQ, left, right, 0); // Expression (pLeft = pRight)
                where_ = Expr.And(ctx, where_, eq);
            }

            // If the child table is the same as the parent table, and this scan is taking place as part of a DELETE operation (operation D.2), omit the
            // row being deleted from the scan by adding ($rowid != rowid) to the WHERE clause, where $rowid is the rowid of the row being deleted.
            if (table == fkey.From && incr > 0)
            {
                Expr left = Expr.Expr(ctx, TK.REGISTER, null); // Value from parent table row
                Expr right = Expr.Expr(ctx, TK.COLUMN, null); // Column ref to child table
                if (left != null && right != null)
                {
                    left.TableId = regDataId;
                    left.Aff = AFF.INTEGER;
                    right.TableId = src.Ids[0].Cursor;
                    right.ColumnIdx = -1;
                }
                Expr eq = Expr.PExpr(parse, TK.NE, left, right, 0); // Expression (pLeft = pRight)
                where_ = Expr.And(ctx, where_, eq);
            }

            // Resolve the references in the WHERE clause.
            NameContext nameContext; // Context used to resolve WHERE clause
            nameContext = new NameContext(); // memset( &sNameContext, 0, sizeof( NameContext ) );
            nameContext.SrcList = src;
            nameContext.Parse = parse;
            ResolveExprNames(nameContext, ref where_);

            // Create VDBE to loop through the entries in src that match the WHERE clause. If the constraint is not deferred, throw an exception for
            // each row found. Otherwise, for deferred constraints, increment the deferred constraint counter by incr for each row selected.
            ExprList dummy = null;
            WhereInfo whereInfo = Where.Begin(parse, src, where_, ref dummy, 0); // Context used by sqlite3WhereXXX()
            if (incr > 0 && !fkey.IsDeferred)
                E.Parse_Toplevel(parse).MayAbort = true;
            v.AddOp2(OP.FkCounter, fkey.IsDeferred, incr);
            if (whereInfo != null)
                Where.End(whereInfo);

            // Clean up the WHERE clause constructed above.
            Expr.Delete(ctx, ref where_);
            if (fkIfZero != 0)
                v.JumpHere(fkIfZero);
        }

        public static FKey FKReferences(Table table)
        {
            int nameLength = table.Name.Length;
            return table.Schema.FKeyHash.Find(table.Name, nameLength, (FKey)null);
        }

        static void FKTriggerDelete(Context ctx, Trigger p)
        {
            if (p != null)
            {
                TriggerStep step = p.StepList;
                Expr.Delete(ctx, ref step.Where);
                Expr.ListDelete(ctx, ref step.ExprList);
                Select.Delete(ctx, ref step.Select);
                Expr.Delete(ctx, ref p.When);
                C._tagfree(ctx, ref p);
            }
        }

        public void FKDropTable(SrcList name, Table table)
        {
            Context ctx = Ctx;
            if ((ctx.Flags & Context.FLAG.ForeignKeys) != 0 && !IsVirtual(table) && table.Select == null)
            {
                int skipId = 0;
                Vdbe v = GetVdbe();

                Debug.Assert(v != null); // VDBE has already been allocated
                if (FKReferences(table) == null)
                {
                    // Search for a deferred foreign key constraint for which this table is the child table. If one cannot be found, return without 
                    // generating any VDBE code. If one can be found, then jump over the entire DELETE if there are no outstanding deferred constraints
                    // when this statement is run.
                    FKey p;
                    for (p = table.FKeys; p != null; p = p.NextFrom)
                        if (p.IsDeferred) break;
                    if (p == null) return;
                    skipId = v.MakeLabel();
                    v.AddOp2(OP.FkIfZero, 1, skipId);
                }

                DisableTriggers = true;
                DeleteFrom(this, SrcListDup(ctx, name, 0), null);
                DisableTriggers = false;

                // If the DELETE has generated immediate foreign key constraint violations, halt the VDBE and return an error at this point, before
                // any modifications to the schema are made. This is because statement transactions are not able to rollback schema changes.
                v.AddOp2(OP.FkIfZero, 0, v.CurrentAddr() + 2);
                HaltConstraint(this, OE.Abort, "foreign key constraint failed", Vdbe.P4T.STATIC);

                if (skipId != 0)
                    v.ResolveLabel(skipId);
            }
        }

        public void FKCheck(Table table, int regOld, int regNew)
        {
            Context ctx = Ctx; // Database handle
            bool isIgnoreErrors = DisableTriggers;

            // Exactly one of regOld and regNew should be non-zero.
            Debug.Assert((regOld == 0) != (regNew == 0));

            // If foreign-keys are disabled, this function is a no-op.
            if ((ctx.Flags & Context.FLAG.ForeignKeys) == 0) return;

            int db = SchemaToIndex(ctx, table.Schema); // Index of database containing pTab
            string dbName = ctx.DBs[db].Name; // Name of database containing pTab

            // Loop through all the foreign key constraints for which pTab is the child table (the table that the foreign key definition is part of).
            FKey fkey; // Used to iterate through FKs
            for (fkey = table.FKeys; fkey != null; fkey = fkey.NextFrom)
            {
                bool isIgnore = false;
                // Find the parent table of this foreign key. Also find a unique index on the parent key columns in the parent table. If either of these 
                // schema items cannot be located, set an error in pParse and return early.
                Table to = (DisableTriggers ? FindTable(ctx, fkey.To, dbName) : LocateTable(false, fkey.To, dbName)); // Parent table of foreign key pFKey
                Index index = null; // Index on key columns in pTo
                int[] frees = null;
                if (to == null || LocateFkeyIndex(to, fkey, out index, out frees) != 0)
                {
                    Debug.Assert(!isIgnoreErrors || (regOld != 0 && regNew == 0));
                    if (!isIgnoreErrors || ctx.MallocFailed) return;
                    if (to == null)
                    {
                        // If isIgnoreErrors is true, then a table is being dropped. In this se SQLite runs a "DELETE FROM xxx" on the table being dropped
                        // before actually dropping it in order to check FK constraints. If the parent table of an FK constraint on the current table is
                        // missing, behave as if it is empty. i.e. decrement the FK counter for each row of the current table with non-NULL keys.
                        Vdbe v = GetVdbe();
                        int jumpId = v.CurrentAddr() + fkey.Cols.length + 1;
                        for (int i = 0; i < fkey.Cols.length; i++)
                        {
                            int regId = fkey.Cols[i].From + regOld + 1;
                            v.AddOp2(OP.IsNull, regId, jumpId);
                        }
                        v.AddOp2(OP.FkCounter, fkey.IsDeferred, -1);
                    }
                    continue;
                }
                Debug.Assert(fkey.Cols.length == 1 || (frees != null && index != null));

                int[] cols;
                if (frees != null)
                    cols = frees;
                else
                {
                    int col = fkey.Cols[0].From;
                    cols = new int[1];
                    cols[0] = col;
                }
                for (int i = 0; i < fkey.Cols.length; i++)
                {
                    if (cols[i] == table.PKey)
                        cols[i] = -1;
#if !OMIT_AUTHORIZATION
                    // Request permission to read the parent key columns. If the authorization callback returns SQLITE_IGNORE, behave as if any
                    // values read from the parent table are NULL.
                    if (ctx.Auth != null)
                    {
                        string colName = to.Cols[index != null ? index.Columns[i] : to.PKey].Name;
                        ARC rcauth = Auth.ReadColumn(this, to.Name, colName, db);
                        isIgnore = (rcauth == ARC.IGNORE);
                    }
#endif
                }

                // Take a shared-cache advisory read-lock on the parent table. Allocate a cursor to use to search the unique index on the parent key columns 
                // in the parent table.
                TableLock(db, to.Id, false, to.Name);
                Tabs++;

                if (regOld != 0) // A row is being removed from the child table. Search for the parent. If the parent does not exist, removing the child row resolves an outstanding foreign key constraint violation.
                    FKLookupParent(this, db, to, index, fkey, cols, regOld, -1, isIgnore);
                if (regNew != 0) // A row is being added to the child table. If a parent row cannot be found, adding the child row has violated the FK constraint. 
                    FKLookupParent(this, db, to, index, fkey, cols, regNew, +1, isIgnore);

                C._tagfree(ctx, ref frees);
            }

            // Loop through all the foreign key constraints that refer to this table
            for (fkey = FKReferences(table); fkey != null; fkey = fkey.NextTo)
            {
                if (!fkey.IsDeferred && Toplevel == null && !IsMultiWrite)
                {
                    Debug.Assert(regOld == 0 && regNew != 0);
                    // Inserting a single row into a parent table cannot cause an immediate foreign key violation. So do nothing in this case.
                    continue;
                }

                Index index = null; // Foreign key index for pFKey
                int[] cols = null;
                if (LocateFkeyIndex(table, fkey, out index, out cols) != 0)
                {
                    if (isIgnoreErrors || ctx.MallocFailed) return;
                    continue;
                }
                Debug.Assert(cols != null || fkey.Cols.length == 1);

                // Create a SrcList structure containing a single table (the table the foreign key that refers to this table is attached to). This
                // is required for the sqlite3WhereXXX() interface.
                SrcList src = SrcListAppend(ctx, null, null, null);
                if (src != null)
                {
                    SrcList.SrcListItem item = src.Ids[0];
                    item.Table = fkey.From;
                    item.Name = fkey.From.Name;
                    item.Table.Refs++;
                    item.Cursor = Tabs++;

                    if (regNew != 0)
                        FKScanChildren(this, src, table, index, fkey, cols, regNew, -1);
                    if (regOld != 0)
                    {
                        // If there is a RESTRICT action configured for the current operation on the parent table of this FK, then throw an exception 
                        // immediately if the FK constraint is violated, even if this is a deferred trigger. That's what RESTRICT means. To defer checking
                        // the constraint, the FK should specify NO ACTION (represented using OE_None). NO ACTION is the default.
                        FKScanChildren(this, src, table, index, fkey, cols, regOld, 1);
                    }
                    item.Name = null;
                    SrcListDelete(ctx, ref src);
                }
                C._tagfree(ctx, ref cols);
            }
        }

        static uint COLUMN_MASK(int x) { return ((x) > 31) ? 0xffffffff : ((uint)1 << (x)); } //: #define COLUMN_MASK(x) (((x)>31) ? 0xffffffff : ((uint32)1<<(x)))

        public uint FKOldmask(Table table)
        {
            uint mask = 0;
            if ((Ctx.Flags & Context.FLAG.ForeignKeys) != 0)
            {
                FKey p;
                int i;
                for (p = table.FKeys; p != null; p = p.NextFrom)
                    for (i = 0; i < p.Cols.length; i++) mask |= COLUMN_MASK(p.Cols[i].From);
                for (p = FKReferences(table); p != null; p = p.NextTo)
                {
                    Index index;
                    int[] dummy;
                    LocateFkeyIndex(table, p, out index, out dummy);
                    if (index != null)
                        for (i = 0; i < index.Columns.length; i++) mask |= COLUMN_MASK(index.Columns[i]);
                }
            }
            return mask;
        }

        public bool FKRequired(Table table, int[] changes, int chngRowid)
        {
            if ((Ctx.Flags & Context.FLAG.ForeignKeys) != 0)
            {
                if (changes == null)  // A DELETE operation. Foreign key processing is required if the table in question is either the child or parent table for any foreign key constraint.
                    return (FKReferences(table) != null || table.FKeys != null);
                else // This is an UPDATE. Foreign key processing is only required if operation modifies one or more child or parent key columns.
                {
                    int i;
                    FKey p;
                    // Check if any child key columns are being modified.
                    for (p = table.FKeys; p != null; p = p.NextFrom)
                    {
                        for (i = 0; i < p.Cols.length; i++)
                        {
                            int childKeyId = p.Cols[i].From;
                            if (changes[childKeyId] >= 0) return true;
                            if (childKeyId == table.PKey && chngRowid != 0) return true;
                        }
                    }

                    // Check if any parent key columns are being modified.
                    for (p = FKReferences(table); p != null; p = p.NextTo)
                        for (i = 0; i < p.Cols.length; i++)
                        {
                            string keyName = p.Cols[i].Col;
                            for (int key = 0; key < table.Cols.length; key++)
                            {
                                Column col = table.Cols[key];
                                if (!string.IsNullOrEmpty(keyName) ? string.Equals(col.Name, keyName, StringComparison.InvariantCultureIgnoreCase) : (col.ColFlags & COLFLAG.PRIMKEY) != 0)
                                {
                                    if (changes[key] >= 0) return true;
                                    if (key == table.PKey && chngRowid != 0) return true;
                                }
                            }
                        }
                }
            }
            return false;
        }

        static Trigger FKActionTrigger(Parse parse, Table table, FKey fkey, ExprList changes)
        {
            Context ctx = parse.Ctx; // Database handle
            int actionId = (changes != null ? 1 : 0);  // 1 for UPDATE, 0 for DELETE
            OE action = fkey.Actions[actionId]; // One of OE_None, OE_Cascade etc.
            Trigger trigger = fkey.Triggers[actionId]; // Trigger definition to return

            if (action != OE.None && trigger == null)
            {

                Index index = null; // Parent key index for this FK
                int[] cols = null; // child table cols . parent key cols
                if (LocateFkeyIndex(parse, table, fkey, out index, out cols) != 0) return null;
                Debug.Assert(cols != null || fkey.Cols.length == 1);

                Expr where_ = null; // WHERE clause of trigger step
                Expr when = null; // WHEN clause for the trigger
                ExprList list = null; // Changes list if ON UPDATE CASCADE
                for (int i = 0; i < fkey.Cols.length; i++)
                {
                    Token oldToken = new Token("old", 3); // Literal "old" token
                    Token newToken = new Token("new", 3); // Literal "new" token

                    int fromColId = (cols != null ? cols[i] : fkey.Cols[0].From); // Idx of column in child table
                    Debug.Assert(fromColId >= 0);
                    Token fromCol = new Token(); // Name of column in child table
                    Token toCol = new Token(); // Name of column in parent table
                    toCol.data = (index != null ? table.Cols[index.Columns[i]].Name : "oid");
                    fromCol.data = fkey.From.Cols[fromColId].Name;
                    toCol.length = (uint)toCol.data.Length;
                    fromCol.length = (uint)fromCol.data.Length;

                    // Create the expression "OLD.zToCol = zFromCol". It is important that the "OLD.zToCol" term is on the LHS of the = operator, so
                    // that the affinity and collation sequence associated with the parent table are used for the comparison.
                    Expr eq = Expr.PExpr(parse, TK.EQ,
                        Expr.PExpr(parse, TK.DOT,
                            Expr.PExpr(parse, TK.ID, null, null, oldToken),
                            Expr.PExpr(parse, TK.ID, null, null, toCol)
                            , 0),
                        Expr.PExpr(parse, TK.ID, null, null, fromCol)
                        , 0); // tFromCol = OLD.tToCol
                    where_ = Expr.And(ctx, where_, eq);

                    // For ON UPDATE, construct the next term of the WHEN clause. The final WHEN clause will be like this:
                    //
                    //    WHEN NOT(old.col1 IS new.col1 AND ... AND old.colN IS new.colN)
                    if (changes != null)
                    {
                        eq = Expr.PExpr(parse, TK.IS,
                            Expr.PExpr(parse, TK.DOT,
                            Expr.PExpr(parse, TK.ID, null, null, oldToken),
                            Expr.PExpr(parse, TK.ID, null, null, toCol),
                            0),
                            Expr.PExpr(parse, TK.DOT,
                            Expr.PExpr(parse, TK.ID, null, null, newToken),
                            Expr.PExpr(parse, TK.ID, null, null, toCol),
                            0),
                            0);
                        when = Expr.And(ctx, when, eq);
                    }

                    if (action != OE.Restrict && (action != OE.Cascade || changes != null))
                    {
                        Expr newExpr;
                        if (action == OE.Cascade)
                            newExpr = Expr.PExpr(parse, TK.DOT,
                                Expr.PExpr(parse, TK.ID, null, null, newToken),
                                Expr.PExpr(parse, TK.ID, null, null, toCol)
                                , 0);
                        else if (action == OE.SetDflt)
                        {
                            Expr dfltExpr = fkey.From.Cols[fromColId].Dflt;
                            if (dfltExpr != null)
                                newExpr = Expr.Dup(ctx, dfltExpr, 0);
                            else
                                newExpr = Expr.PExpr(parse, TK.NULL, 0, 0, 0);
                        }
                        else
                            newExpr = Expr.PExpr(parse, TK.NULL, 0, 0, 0);
                        list = Expr.ListAppend(parse, list, newExpr);
                        Expr.ListSetName(parse, list, fromCol, 0);
                    }
                }
                C._tagfree(ctx, ref cols);

                string fromName = fkey.From.Name; // Name of child table
                int fromNameLength = fromName.Length; // Length in bytes of zFrom

                Select select = null; // If RESTRICT, "SELECT RAISE(...)"
                if (action == OE.Restrict)
                {
                    Token from = new Token();
                    from.data = fromName;
                    from.length = fromNameLength;
                    Expr raise = Expr.Expr(ctx, TK.RAISE, "foreign key constraint failed");
                    if (raise != null)
                        raise.Affinity = OE.Abort;
                    select = Select.New(parse,
                        Expr.ListAppend(parse, 0, raise),
                        SrcListAppend(ctx, 0, from, null),
                        where_,
                        null, null, null, 0, null, null);
                    where_ = null;
                }

                // Disable lookaside memory allocation
                bool enableLookaside = ctx.Lookaside.Enabled; // Copy of ctx->lookaside.bEnabled
                ctx.Lookaside.Enabled = false;

                trigger = new Trigger();
                //: trigger = (Trigger *)_tagalloc(ctx, 
                //:    sizeof(Trigger) + // Trigger
                //:    sizeof(TriggerStep) + // Single step in trigger program
                //:    fromNameLength + 1 // Space for pStep->target.z
                //:    , true);
                TriggerStep step = null; // First (only) step of trigger program                
                if (trigger != null)
                {
                    step = trigger.StepList = new TriggerStep(); //: (TriggerStep)trigger[1];
                    step.Target.data = fromName; //: (char *)&step[1];
                    step.Target.length = fromNameLength;
                    //: _memcpy((const char *)step->Target.data, fromName, fromNameLength);

                    step.Where = Expr.Dup(ctx, where_, EXPRDUP_REDUCE);
                    step.ExprList = Expr.ListDup(ctx, list, EXPRDUP_REDUCE);
                    step.Select = Select.Dup(ctx, select, EXPRDUP_REDUCE);
                    if (when != null)
                    {
                        when = Expr.PExpr(parse, TK.NOT, when, 0, 0);
                        trigger.When = Expr.Dup(ctx, when, EXPRDUP_REDUCE);
                    }
                }

                // Re-enable the lookaside buffer, if it was disabled earlier.
                ctx.Lookaside.Enabled = enableLookaside;

                Expr.Delete(ctx, ref where_);
                Expr.Delete(ctx, ref when);
                Expr.ListDelete(ctx, ref list);
                Select.Delete(ctx, ref select);
                if (ctx.MallocFailed)
                {
                    FKTriggerDelete(ctx, trigger);
                    return null;
                }

                switch (action)
                {
                    case OE.Restrict:
                        step.OP = TK.SELECT;
                        break;
                    case OE.Cascade:
                        if (changes == null)
                        {
                            step.OP = TK.DELETE;
                            break;
                        }
                        goto default;
                    default:
                        step.OP = TK.UPDATE;
                        break;
                }
                step.Trigger = trigger;
                trigger.Schema = table.Schema;
                trigger.TabSchema = table.Schema;
                fkey.Triggers[actionId] = trigger;
                trigger.OP = (TK)(changes != null ? TK.UPDATE : TK.DELETE);
            }

            return trigger;
        }

        public void FKActions(Table table, ExprList changes, int regOld)
        {
            // If foreign-key support is enabled, iterate through all FKs that refer to table table. If there is an action associated with the FK 
            // for this operation (either update or delete), invoke the associated trigger sub-program.
            if ((Ctx.Flags & Context.FLAG.ForeignKeys) != 0)
                for (FKey fkey = FKReferences(table); fkey != null; fkey = fkey.NextTo)
                {
                    Trigger action = fkActionTrigger(table, fkey, changes);
                    if (action != null)
                        CodeRowTriggerDirect(this, action, table, regOld, OE.Abort, 0);
                }
        }

#endif

        public static void FKDelete(Context ctx, Table table)
        {
            Debug.Assert(ctx == null || Btree.SchemaMutexHeld(ctx, 0, table.Schema));
            FKey next; // Copy of pFKey.pNextFrom
            for (FKey fkey = table.FKeys; fkey != null; fkey = next)
            {
                // Remove the FK from the fkeyHash hash table.
                //: if (!ctx || ctx->BytesFreed == 0)
                {
                    if (fkey.PrevTo != null)
                        fkey.PrevTo.NextTo = fkey.NextTo;
                    else
                    {
                        FKey p = fkey.NextTo;
                        string z = (p != null ? fkey.NextTo.To : fkey.To);
                        table.Schema.FKeyHash.Insert(z, z.Length, p);
                    }
                    if (fkey.NextTo != null)
                        fkey.NextTo.PrevTo = fkey.PrevTo;
                }

                // EV: R-30323-21917 Each foreign key constraint in SQLite is classified as either immediate or deferred.
                Debug.Assert(fkey.IsDeferred == false || fkey.IsDeferred == true);

                // Delete any triggers created to implement actions for this FK.
#if !OMIT_TRIGGER
                FKTriggerDelete(ctx, fkey.Triggers[0]);
                FKTriggerDelete(ctx, fkey.Triggers[1]);
#endif
                next = fkey.NextFrom;
                C._tagfree(ctx, ref fkey);
            }
        }
    }
}
#endif