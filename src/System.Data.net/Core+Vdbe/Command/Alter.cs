#region OMIT_ALTERTABLE
#if !OMIT_ALTERTABLE
using System;
using System.Diagnostics;
using System.Text;

namespace Core.Command
{
    public class Alter
    {
        static void RenameTableFunc(FuncContext fctx, int notUsed, Mem[] argv)
        {
            Context ctx = Vdbe.Context_Ctx(fctx);
            string sql = Vdbe.Value_Text(argv[0]);
            string tableName = Vdbe.Value_Text(argv[1]);
            if (string.IsNullOrEmpty(sql))
                return;
            int length = 0;
            TK token = 0;
            Token tname = new Token();

            int z = 0, zLoc = 0;

            // The principle used to locate the table name in the CREATE TABLE statement is that the table name is the first non-space token that
            // is immediately followed by a TK_LP or TK_USING token.
            do
            {
                if (z == sql.Length)
                    return; // Ran out of input before finding an opening bracket. Return NULL.
                // Store the token that zCsr points to in tname.
                zLoc = z;
                tname.data = sql.Substring(z);
                tname.length = (uint)length;

                // Advance zCsr to the next token. Store that token type in 'token', and its length in 'len' (to be used next iteration of this loop).
                do
                {
                    z += length;
                    length = (z == sql.Length ? 1 : Parse.GetToken(sql, z, ref token));
                } while (token == TK.SPACE);
                Debug.Assert(length > 0);
            } while (token != TK.LP && token != TK.USING);

            string r = C._mtagprintf(ctx, "%.*s\"%w\"%s", zLoc, sql.Substring(0, zLoc), tableName, sql.Substring(zLoc + (int)tname.length));
            Vdbe.Result_Text(fctx, r, -1, DESTRUCTOR_DYNAMIC);
        }

#if !OMIT_FOREIGN_KEY
        static void RenameParentFunc(FuncContext fctx, int notUsed, Mem[] argv)
        {
            Context ctx = Vdbe.Context_Ctx(fctx);
            string input = Vdbe.Value_Text(argv[0]);
            string oldName = Vdbe.Value_Text(argv[1]);
            string newName = Vdbe.Value_Text(argv[2]);

            int zIdx;         // Pointer to token
            int zLeft = 0;    // Pointer to remainder of String

            TK token = 0;    // Type of token

            string output = string.Empty;
            int n; // Length of token z
            for (int z = 0; z < input.Length; z += n)
            {
                n = Parse.GetToken(input, z, ref token);
                if (token == TK.REFERENCES)
                {
                    string parent;
                    do
                    {
                        z += n;
                        n = Parse.GetToken(input, z, ref token);
                    } while (token == TK.SPACE);

                    parent = (z + n < input.Length ? input.Substring(z, n) : string.Empty);
                    if (string.IsNullOrEmpty(parent)) break;
                    Parse.Dequote(ref parent);
                    if (oldName.Equals(parent, StringComparison.OrdinalIgnoreCase))
                    {
                        string out_ = C._mtagprintf(ctx, "%s%.*s\"%w\"", output, z - zLeft, input.Substring(zLeft), newName);
                        C._tagfree(ctx, ref output);
                        output = out_;
                        z += n;
                        zLeft = z;
                    }
                    C._tagfree(ctx, ref parent);
                }
            }

            string r = C._mtagprintf(ctx, "%s%s", output, input.Substring(zLeft));
            Vdbe.Result_Text(fctx, r, -1, DESTRUCTOR.DYNAMIC);
            C._tagfree(ctx, ref output);
        }
#endif

#if !OMIT_TRIGGER
        static void RenameTriggerFunc(FuncContext fctx, int notUsed, Mem[] argv)
        {
            Context ctx = Vdbe.Context_Ctx(fctx);
            string sql = Vdbe.Value_Text(argv[0]);
            string tableName = Vdbe.Value_Text(argv[1]);

            int z = 0, zLoc = 0;
            int length = 1;
            TK token = 0;
            Token tname = new Token();
            int dist = 3;

            // The principle used to locate the table name in the CREATE TRIGGER statement is that the table name is the first token that is immediatedly
            // preceded by either TK_ON or TK_DOT and immediatedly followed by one of TK_WHEN, TK_BEGIN or TK_FOR.
            if (sql != null)
                return;
            do
            {
                if (z == sql.Length)
                    return; // Ran out of input before finding the table name. Return NULL.

                // Store the token that zCsr points to in tname.
                zLoc = z;
                tname.data = sql.Substring(z, length);
                tname.length = (uint)length;

                // Advance zCsr to the next token. Store that token type in 'token', and its length in 'len' (to be used next iteration of this loop).
                do
                {
                    z += length;
                    length = (z == sql.Length ? 1 : Parse.GetToken(sql, z, ref token));
                } while (token == TK.SPACE);
                Debug.Assert(length > 0);

                // Variable 'dist' stores the number of tokens read since the most recent TK_DOT or TK_ON. This means that when a WHEN, FOR or BEGIN 
                // token is read and 'dist' equals 2, the condition stated above to be met.
                //
                // Note that ON cannot be a database, table or column name, so there is no need to worry about syntax like 
                // "CREATE TRIGGER ... ON ON.ON BEGIN ..." etc.

                dist++;
                if (token == TK.DOT || token == TK.ON)
                    dist = 0;
            } while (dist != 2 || (token != TK.WHEN && token != TK.FOR && token != TK.BEGIN));

            // Variable tname now contains the token that is the old table-name in the CREATE TRIGGER statement.
            string r = C._mtagprintf(ctx, "%.*s\"%w\"%s", zLoc, sql.Substring(0, zLoc), tableName, sql.Substring(zLoc + (int)tname.length));
            Vdbe.Result_Text(fctx, r, -1, C.DESTRUCTOR_DYNAMIC);
        }
#endif

        static FuncDef[] _alterTableFuncs = new FuncDef[] {
            FUNCTION("sqlite_rename_table",   2, 0, 0, RenameTableFunc),
#if !OMIT_TRIGGER
            FUNCTION("sqlite_rename_trigger", 2, 0, 0, RenameTriggerFunc),
#endif
#if !OMIT_FOREIGN_KEY
            FUNCTION("sqlite_rename_parent",  3, 0, 0, RenameParentFunc),
#endif
  };
        public static void Functions()
        {
            FuncDefHash hash = Main.GlobalFunctions;
            FuncDef[] funcs = _alterTableFuncs;
            for (int i = 0; i < _alterTableFuncs.Length; i++)
                Callback.FuncDefInsert(hash, funcs[i]);
        }

        static string WhereOrName(Context ctx, string where_, string constant)
        {
            string newExpr;
            if (string.IsNullOrEmpty(where_))
                newExpr = C._mtagprintf(ctx, "name=%Q", constant);
            else
            {
                newExpr = C._mtagprintf(ctx, "%s OR name=%Q", where_, constant);
                C._tagfree(ctx, ref where_);
            }
            return newExpr;
        }

#if !OMIT_FOREIGN_KEY&& !OMIT_TRIGGER
        static string WhereForeignKeys(Parse parse, Table table)
        {
            string where_ = string.Empty;
            for (FKey p = Parse.FKReferences(table); p != null; p = p.NextTo)
                where_ = WhereOrName(parse.Ctx, where_, p.From.Name);
            return where_;
        }
#endif

        static string WhereTempTriggers(Parse parse, Table table)
        {
            Context ctx = parse.Ctx;
            string where_ = string.Empty;
            Schema tempSchema = ctx.DBs[1].Schema; // Temp db schema
            // If the table is not located in the temp.db (in which case NULL is returned, loop through the tables list of triggers. For each trigger
            // that is not part of the temp.db schema, add a clause to the WHERE expression being built up in zWhere.
            if (table.Schema != tempSchema)
                for (Trigger trig = Trigger.List(parse, table); trig != null; trig = trig.Next)
                    if (trig.Schema == tempSchema)
                        where_ = WhereOrName(ctx, where_, trig.Name);
            if (!string.IsNullOrEmpty(where_))
                where_ = C._mtagprintf(ctx, "type='trigger' AND (%s)", where_);
            return where_;
        }

        static void ReloadTableSchema(Parse parse, Table table, string name)
        {
            Context ctx = parse.Ctx;
            string where_;
#if !SQLITE_OMIT_TRIGGER
            Trigger trig;
#endif

            Vdbe v = parse.GetVdbe();
            if (C._NEVER(v == null)) return;
            Debug.Assert(Btree.HoldsAllMutexes(ctx));
            int db = Prepare.SchemaToIndex(ctx, table.Schema); // Index of database containing pTab
            Debug.Assert(db >= 0);

#if !OMIT_TRIGGER
            // Drop any table triggers from the internal schema.
            for (trig = Trigger.List(parse, table); trig != null; trig = trig.Next)
            {
                int trigDb = Prepare.SchemaToIndex(ctx, trig.Schema);
                Debug.Assert(trigDb == db || trigDb == 1);
                v.AddOp4(OP.DropTrigger, trigDb, 0, 0, trig.Name, 0);
            }
#endif

            // Drop the table and index from the internal schema.
            v.AddOp4(OP.DropTable, db, 0, 0, table.Name, 0);

            // Reload the table, index and permanent trigger schemas.
            where_ = C._mtagprintf(ctx, "tbl_name=%Q", name);
            if (where_ == null) return;
            v.AddParseSchemaOp(db, where_);

#if !OMIT_TRIGGER
            // Now, if the table is not stored in the temp database, reload any temp triggers. Don't use IN(...) in case SQLITE_OMIT_SUBQUERY is defined.
            if ((where_ = WhereTempTriggers(parse, table)) != "")
                v.AddParseSchemaOp(1, where_);
#endif
        }

        static bool IsSystemTable(Parse parse, string name)
        {
            if (name.StartsWith("sqlite_", StringComparison.OrdinalIgnoreCase))
            {
                parse.ErrorMsg("table %s may not be altered", name);
                return true;
            }
            return false;
        }

        public static void RenameTable(Parse parse, SrcList src, Token name)
        {
            Context ctx = parse.Ctx; // Database connection

            Context.FLAG savedDbFlags = ctx.Flags; // Saved value of db->flags
            //if (C._NEVER(ctx.MallocFailed)) goto exit_rename_table;
            Debug.Assert(src.Srcs == 1);
            Debug.Assert(Btree.HoldsAllMutexes(ctx));

            Table table = parse.LocateTableItem(false, src.Ids[0]); // Table being renamed
            if (table == null) goto exit_rename_table;
            int db = Prepare.SchemaToIndex(ctx, table.Schema); // Database that contains the table
            string dbName = ctx.DBs[db].Name; // Name of database iDb
            ctx.Flags |= Context.FLAG.PreferBuiltin;

            // Get a NULL terminated version of the new table name.
            string nameAsString = Parse.NameFromToken(ctx, name); // NULL-terminated version of pName
            if (nameAsString == null) goto exit_rename_table;

            // Check that a table or index named 'zName' does not already exist in database iDb. If so, this is an error.
            if (Parse.FindTable(ctx, nameAsString, dbName) != null || Parse.FindIndex(ctx, nameAsString, dbName) != null)
            {
                parse.ErrorMsg("there is already another table or index with this name: %s", nameAsString);
                goto exit_rename_table;
            }

            // Make sure it is not a system table being altered, or a reserved name that the table is being renamed to.
            if (IsSystemTable(parse, table.Name) || parse->CheckObjectName(nameAsString) != RC.OK)
                goto exit_rename_table;

#if !OMIT_VIEW
            if (table.Select != null)
            {
                parse.ErrorMsg("view %s may not be altered", table.Name);
                goto exit_rename_table;
            }
#endif

#if !OMIT_AUTHORIZATION
            // Invoke the authorization callback.
            if (Auth.Check(parse, AUTH.ALTER_TABLE, dbName, table.Name, null))
                goto exit_rename_table;
#endif

            VTable vtable = null; // Non-zero if this is a v-tab with an xRename()
#if !OMIT_VIRTUALTABLE
            if (parse->ViewGetColumnNames(table) != 0)
                goto exit_rename_table;
            if (E.IsVirtual(table))
            {
                vtable = VTable.GetVTable(ctx, table);
                if (vtable.IVTable.IModule.Rename == null)
                    vtable = null;
            }
#endif

            // Begin a transaction and code the VerifyCookie for database iDb. Then modify the schema cookie (since the ALTER TABLE modifies the
            // schema). Open a statement transaction if the table is a virtual table.
            Vdbe v = parse.GetVdbe();
            if (v == null) goto exit_rename_table;
            parse.BeginWriteOperation((vtable != null ? 1 : 0), db);
            parse.ChangeCookie(db);

            // If this is a virtual table, invoke the xRename() function if one is defined. The xRename() callback will modify the names
            // of any resources used by the v-table implementation (including other SQLite tables) that are identified by the name of the virtual table.
#if  !OMIT_VIRTUALTABLE
            if (vtable != null)
            {
                int i = ++parse.Mems;
                v.AddOp4(OP.String8, 0, i, 0, nameAsString, 0);
                v.AddOp4(OP.VRename, i, 0, 0, vtable, Vdbe.P4T.VTAB);
                parse.MayAbort();
            }
#endif

            // figure out how many UTF-8 characters are in zName
            string tableName = table.Name; // Original name of the table
            int tableNameLength = C._utf8charlength(tableName, -1); // Number of UTF-8 characters in zTabName

#if !OMIT_TRIGGER
            string where_ = string.Empty; // Where clause to locate temp triggers
#endif

#if !OMIT_FOREIGN_KEY && !OMIT_TRIGGER
            if ((ctx.Flags & Context.FLAG.ForeignKeys) != 0)
            {
                // If foreign-key support is enabled, rewrite the CREATE TABLE statements corresponding to all child tables of foreign key constraints
                // for which the renamed table is the parent table.
                if ((where_ = WhereForeignKeys(parse, table)) != null)
                {
                    parse.NestedParse(
                        "UPDATE \"%w\".%s SET " +
                            "sql = sqlite_rename_parent(sql, %Q, %Q) " +
                            "WHERE %s;", dbName, E.SCHEMA_TABLE(db), tableName, nameAsString, where_);
                    C._tagfree(ctx, ref where_);
                }
            }
#endif

            // Modify the sqlite_master table to use the new table name.
            parse.NestedParse(
            "UPDATE %Q.%s SET " +
#if OMIT_TRIGGER
 "sql = sqlite_rename_table(sql, %Q), " +
#else
 "sql = CASE " +
            "WHEN type = 'trigger' THEN sqlite_rename_trigger(sql, %Q)" +
            "ELSE sqlite_rename_table(sql, %Q) END, " +
#endif
 "tbl_name = %Q, " +
            "name = CASE " +
            "WHEN type='table' THEN %Q " +
            "WHEN name LIKE 'sqlite_autoindex%%' AND type='index' THEN " +
            "'sqlite_autoindex_' || %Q || substr(name,%d+18) " +
            "ELSE name END " +
            "WHERE tbl_name=%Q AND " +
            "(type='table' OR type='index' OR type='trigger');",
            dbName, SCHEMA_TABLE(db), nameAsString, nameAsString, nameAsString,
#if !OMIT_TRIGGER
 nameAsString,
#endif
 nameAsString, tableNameLength, tableName);

#if !OMIT_AUTOINCREMENT
            // If the sqlite_sequence table exists in this database, then update it with the new table name.
            if (Parse.FindTable(ctx, "sqlite_sequence", dbName) != null)
                parse.NestedParse(
                "UPDATE \"%w\".sqlite_sequence set name = %Q WHERE name = %Q",
                dbName, nameAsString, table.Name);
#endif

#if !OMIT_TRIGGER
            // If there are TEMP triggers on this table, modify the sqlite_temp_master table. Don't do this if the table being ALTERed is itself located in the temp database.
            if ((where_ = WhereTempTriggers(parse, table)) != "")
            {
                parse.NestedParse(
                "UPDATE sqlite_temp_master SET " +
                "sql = sqlite_rename_trigger(sql, %Q), " +
                "tbl_name = %Q " +
                "WHERE %s;", nameAsString, nameAsString, where_);
                C._tagfree(ctx, ref where_);
            }
#endif

#if !(OMIT_FOREIGN_KEY) && !(OMIT_TRIGGER)
            if ((ctx.Flags & Context.FLAG.ForeignKeys) != 0)
            {
                for (FKey p = Parse.FkReferences(table); p != null; p = p.NextTo)
                {
                    Table from = p.From;
                    if (from != table)
                        ReloadTableSchema(parse, p.From, from.Name);
                }
            }
#endif

            // Drop and reload the internal table schema.
            ReloadTableSchema(parse, table, nameAsString);

        exit_rename_table:
            Expr.SrcListDelete(ctx, ref src);
            C._tagfree(ctx, ref nameAsString);
            ctx.Flags = savedDbFlags;
        }

        public static void MinimumFileFormat(Parse parse, int db, int minFormat)
        {
            Vdbe v = parse.GetVdbe();
            // The VDBE should have been allocated before this routine is called. If that allocation failed, we would have quit before reaching this point
            if (C._ALWAYS(v != null))
            {
                int r1 = Expr.GetTempReg(parse);
                int r2 = Expr.GetTempReg(parse);
                v.AddOp3(OP.ReadCookie, db, r1, Btree.META.FILE_FORMAT);
                v.UsesBtree(db);
                v.AddOp2(OP.Integer, minFormat, r2);
                int j1 = v.AddOp3(OP.Ge, r2, 0, r1);
                v.AddOp3(OP.SetCookie, db, Btree.META.FILE_FORMAT, r2);
                v.JumpHere(j1);
                Expr.ReleaseTempReg(parse, r1);
                Expr.ReleaseTempReg(parse, r2);
            }
        }

        public static void FinishAddColumn(Parse parse, Token colDef)
        {
            Context ctx = parse.Ctx; // The database connection
            if (parse.Errs != 0 || ctx.MallocFailed) return;
            Table newTable = parse.NewTable; // Copy of pParse.pNewTable
            Debug.Assert(newTable != null);
            Debug.Assert(Btree.HoldsAllMutexes(ctx));
            int db = Prepare.SchemaToIndex(ctx, newTable.Schema); // Database number
            string dbName = ctx.DBs[db].Name;// Database name
            string tableName = newTable.Name.Substring(16); // Table name: Skip the "sqlite_altertab_" prefix on the name
            Column col = newTable.Cols[newTable.Cols.length - 1]; // The new column
            Expr dflt = col.Dflt; // Default value for the new column
            Table table = Parse.FindTable(ctx, tableName, dbName); // Table being altered
            Debug.Assert(table != null);

#if !OMIT_AUTHORIZATION
            // Invoke the authorization callback.
            if (Auth.Check(parse, AUTH.ALTER_TABLE, dbName, table.Name, null))
                return;
#endif

            // If the default value for the new column was specified with a literal NULL, then set pDflt to 0. This simplifies checking
            // for an SQL NULL default below.
            if (dflt != null && dflt.OP == TK.NULL)
                dflt = null;

            // Check that the new column is not specified as PRIMARY KEY or UNIQUE. If there is a NOT NULL constraint, then the default value for the
            // column must not be NULL.
            if ((col.ColFlags & COLFLAG.PRIMKEY) != 0)
            {
                parse.ErrorMsg("Cannot add a PRIMARY KEY column");
                return;
            }
            if (newTable.Index != null)
            {
                parse.ErrorMsg("Cannot add a UNIQUE column");
                return;
            }
            if ((ctx.Flags & Context.FLAG.ForeignKeys) != 0 && newTable.FKeys != null && dflt != null)
            {
                parse.ErrorMsg("Cannot add a REFERENCES column with non-NULL default value");
                return;
            }
            if (col.NotNull != 0 && dflt == null)
            {
                parse.ErrorMsg("Cannot add a NOT NULL column with default value NULL");
                return;
            }

            // Ensure the default expression is something that sqlite3ValueFromExpr() can handle (i.e. not CURRENT_TIME etc.)
            if (dflt != null)
            {
                Mem val = null;
                if (Mem_FromExpr(ctx, dflt, TEXTENCODE.UTF8, AFF.NONE, ref val) != 0)
                {
                    ctx.MallocFailed = true;
                    return;
                }
                if (val == null)
                {
                    parse.ErrorMsg("Cannot add a column with non-constant default");
                    return;
                }
                Mem_Free(ref val);
            }

            // Modify the CREATE TABLE statement.
            string colDefAsString = colDef.data.Substring(0, (int)colDef.length).Replace(";", " ").Trim(); // Null-terminated column definition
            if (colDefAsString != null)
            {
                Context.FLAG savedDbFlags = ctx.Flags;
                ctx.Flags |= Context.FLAG.PreferBuiltin;
                parse.NestedParse(
                "UPDATE \"%w\".%s SET " +
                "sql = substr(sql,1,%d) || ', ' || %Q || substr(sql,%d) " +
                "WHERE type = 'table' AND name = %Q",
                dbName, E.SCHEMA_TABLE(db), newTable.AddColOffset, colDefAsString, newTable.AddColOffset + 1,
                tableName);
                C._tagfree(ctx, ref colDefAsString);
                ctx.Flags = savedDbFlags;
            }

            // If the default value of the new column is NULL, then set the file format to 2. If the default value of the new column is not NULL,
            // the file format becomes 3.
            MinimumFileFormat(parse, db, (dflt != null ? 3 : 2));

            // Reload the schema of the modified table.
            ReloadTableSchema(parse, table, table.Name);
        }

        public static void BeginAddColumn(Parse parse, SrcList src)
        {
            Context ctx = parse.Ctx;

            // Look up the table being altered.
            Debug.Assert(parse.NewTable == null);
            Debug.Assert(Btree.HoldsAllMutexes(ctx));
            //if (ctx.MallocFailed) goto exit_begin_add_column;
            Table table = parse.LocateTableItem(false, src.Ids[0]);
            if (table == null) goto exit_begin_add_column;

#if !OMIT_VIRTUALTABLE
            if (IsVirtual(table))
            {
                parse.ErrorMsg("virtual tables may not be altered");
                goto exit_begin_add_column;
            }
#endif

            // Make sure this is not an attempt to ALTER a view.
            if (table.Select != null)
            {
                parse.ErrorMsg("Cannot add a column to a view");
                goto exit_begin_add_column;
            }
            if (IsSystemTable(parse, table.Name))
                goto exit_begin_add_column;

            Debug.Assert(table.AddColOffset > 0);
            int db = Prepare.SchemaToIndex(ctx, table.Schema);

            // Put a copy of the Table struct in Parse.pNewTable for the sqlite3AddColumn() function and friends to modify.  But modify
            // the name by adding an "sqlite_altertab_" prefix.  By adding this prefix, we insure that the name will not collide with an existing
            // table because user table are not allowed to have the "sqlite_" prefix on their name.
            Table newTable = new Table();
            if (newTable == null) goto exit_begin_add_column;
            parse.NewTable = newTable;
            newTable.Refs = 1;
            newTable.Cols.length = table.Cols.length;
            Debug.Assert(newTable.Cols.length > 0);
            int allocs = (((newTable.Cols.length - 1) / 8) * 8) + 8;
            Debug.Assert(allocs >= newTable.Cols.length && allocs % 8 == 0 && allocs - newTable.Cols.length < 8);
            newTable.Cols.data = new Column[allocs];
            newTable.Name = C._mtagprintf(ctx, "sqlite_altertab_%s", table.Name);
            if (newTable.Cols.data == null || newTable.Name == null)
            {
                ctx.MallocFailed = true;
                goto exit_begin_add_column;
            }
            for (int i = 0; i < newTable.Cols.length; i++)
            {
                Column col = table.Cols[i].memcpy();
                col.Coll = null;
                col.Type = null;
                col.Dflt = null;
                col.Dflt = null;
                newTable.Cols[i] = col;
            }
            newTable.Schema = ctx.DBs[db].Schema;
            newTable.AddColOffset = table.AddColOffset;
            newTable.Refs = 1;

            // Begin a transaction and increment the schema cookie.
            parse.BeginWriteOperation(0, db);
            Vdbe v = parse.V;
            if (v == null) goto exit_begin_add_column;
            parse.ChangeCookie(db);

        exit_begin_add_column:
            sqlite3SrcListDelete(ctx, ref src);
            return;
        }
    }
}

#endif
#endregion