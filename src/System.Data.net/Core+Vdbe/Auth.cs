#region OMIT_AUTHORIZATION
#if !OMIT_AUTHORIZATION
using System;
using System.Diagnostics;
using System.Text;
namespace Core
{
    public partial class Auth
    {
        public RC SetAuthorizer(Context ctx, Func<object, int, string, string, string, string, ARC> auth, object arg)
        {
            MutexEx.Enter(ctx.Mutex);
            ctx.Auth = auth;
            ctx.AuthArg = arg;
            Vdbe.ExpirePreparedStatements(ctx);
            MutexEx.Leave(ctx.Mutex);
            return RC.OK;
        }

        public static void BadReturnCode(Parse parse)
        {
            parse.ErrorMsg("authorizer malfunction");
            parse.RC = RC.ERROR;
        }

        public static ARC ReadColumn(Parse parse, string table, string column, int db)
        {
            Context ctx = parse.Ctx; // Database handle
            string dbName = ctx.DBs[db].Name; // Name of attached database
            ARC rc = ctx.Auth(ctx.AuthArg, (int)AUTH.READ, table, column, dbName, parse.AuthContext); // Auth callback return code
            if (rc == ARC.DENY)
            {
                if (ctx.DBs.length > 2 || db != 0)
                    parse.ErrorMsg("access to %s.%s.%s is prohibited", dbName, table, column);
                else
                    parse.ErrorMsg("access to %s.%s is prohibited", table, column);
                parse.RC = RC.AUTH;
            }
            else if (rc != ARC.IGNORE && rc != ARC.OK)
                BadReturnCode(parse);
            return rc;
        }

        public static void Read(Parse parse, Expr expr, Schema schema, SrcList tableList)
        {
            Context ctx = parse.Ctx;
            if (ctx.Auth == null) return;
            int db = Prepare.SchemaToIndex(ctx, schema);// The index of the database the expression refers to
            if (db < 0)
                return; // An attempt to read a column out of a subquery or other temporary table.

            Debug.Assert(expr.OP == TK.COLUMN || expr.OP == TK.TRIGGER);
            Table table = null; // The table being read
            if (expr.OP == TK.TRIGGER)
                table = parse.TriggerTab;
            else
            {
                Debug.Assert(tableList != null);
                for (int src = 0; C._ALWAYS(src < tableList.Srcs); src++)
                    if (expr.TableId == tableList.Ids[src].Cursor)
                    {
                        table = tableList.Ids[src].Table;
                        break;
                    }
            }
            int col = expr.ColumnIdx; // Index of column in table
            if (C._NEVER(table == null)) return;

            string colName; // Name of the column of the table
            if (col >= 0)
            {
                Debug.Assert(col < table.Cols.length);
                colName = table.Cols[col].Name;
            }
            else if (table.PKey >= 0)
            {
                Debug.Assert(table.PKey < table.Cols.length);
                colName = table.Cols[table.PKey].Name;
            }
            else
                colName = "ROWID";
            Debug.Assert(db >= 0 && db < ctx.DBs.length);
            if (ReadColumn(parse, table.Name, colName, db) == ARC.IGNORE)
                expr.OP = TK.NULL;
        }

        public static ARC Check(Parse parse, AUTH code, string arg1, string arg2, string arg3)
        {
            Context ctx = parse.Ctx;
            // Don't do any authorization checks if the database is initialising or if the parser is being invoked from within sqlite3_declare_vtab.
            if (ctx.Init.Busy || E.INDECLARE_VTABLE(parse))
                return ARC.OK;

            if (ctx.Auth == null)
                return ARC.OK;
            ARC rc = ctx.Auth(ctx.AuthArg, code, arg1, arg2, arg3, parse.AuthContext);
            if (rc == ARC.DENY)
            {
                parse.ErrorMsg("not authorized");
                parse.RC = RC.AUTH;
            }
            else if (rc != ARC.OK && rc != ARC.IGNORE)
            {
                rc = ARC.DENY;
                BadReturnCode(parse);
            }
            return rc;
        }

        public void ContextPush(Parse parse, AuthContext actx, string context)
        {
            Debug.Assert(parse != null);
            actx.Parse = parse;
            actx.AuthCtx = parse.AuthContext;
            parse.AuthContext = context;
        }

        public void ContextPop(AuthContext actx)
        {
            if (actx.Parse != null)
            {
                actx.Parse.AuthContext = actx.AuthCtx;
                actx.Parse = null;
            }
        }
    }
}
#endif
#endregion
