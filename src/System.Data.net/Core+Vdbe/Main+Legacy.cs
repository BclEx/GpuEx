using System;
using System.Diagnostics;

namespace Core
{
    public partial class Main
    {
        //C# Alias
        //static public int exec(Context db, string zSql, int NoCallback, int NoArgs, int NoErrors) { string Errors = ""; return sqlite3_exec(db, zSql, null, null, ref Errors); }
        //static public int exec(Context db, string zSql, sqlite3_callback xCallback, object pArg, int NoErrors) { string Errors = ""; return sqlite3_exec(db, zSql, xCallback, pArg, ref Errors); }
        //static public int exec(Context db, string zSql, sqlite3_callback xCallback, object pArg, ref string pzErrMsg) { return sqlite3_exec(db, zSql, xCallback, pArg, ref pzErrMsg); }

        //OVERLOADS 
        public static RC Exec(Context ctx, string sql, int noCallback, int noArgs, int noErrors) { string errors = null; return Exec(ctx, sql, null, null, ref errors); }
        public static RC Exec(Context ctx, string sql, Func<object, int, string[], string[], bool> callback, object arg, int noErrors) { string errors = null; return Exec(ctx, sql, callback, arg, ref errors); }
        public static RC Exec(Context ctx, string sql, Func<object, int, string[], string[], bool> callback, object arg, ref string errmsg)
        {
            RC rc = RC.OK; // Return code
            if (!SafetyCheckOk(ctx)) return SysEx.MISUSE_BKPT();
            if (sql == null) sql = string.Empty;

            MutexEx.Enter(ctx.Mutex);
            Error(ctx, RC.OK, null);
            Vdbe stmt = null; // The current SQL statement
            int retrys = 0; // Number of retry attempts
            string[] colsNames = null; // Names of result columns

            while ((rc == RC.OK || (rc == RC.SCHEMA && (++retrys) < 2)) && sql.Length > 0)
            {

                stmt = null;
                string leftover = null; // Tail of unprocessed SQL
                rc = Prepare.Prepare_(ctx, sql, -1, ref stmt, ref leftover);
                Debug.Assert(rc == RC.OK || stmt == null);
                if (rc != RC.OK)
                    continue;
                if (stmt == null)
                {
                    sql = leftover; // this happens for a comment or white-space
                    continue;
                }

                bool callbackIsInit = false; // True if callback data is initialized
                int cols = Vdbe.Column_Count(stmt);

                while (true)
                {
                    rc = stmt.Step();

                    // Invoke the callback function if required
                    int i;
                    if (callback != null && (rc == RC.ROW || (rc == RC.DONE && !callbackIsInit && (ctx.Flags & Context.FLAG.NullCallback) != 0)))
                    {
                        if (!callbackIsInit)
                        {
                            colsNames = new string[cols];
                            if (colsNames == null)
                                goto exec_out;
                            for (i = 0; i < cols; i++)
                            {
                                colsNames[i] = Vdbe.Column_Name(stmt, i);
                                // Vdbe::SetColName() installs column names as UTF8 strings so there is no way for sqlite3_column_name() to fail.
                                Debug.Assert(colsNames[i] != null);
                            }
                            callbackIsInit = true;
                        }
                        string[] colsValues = null;
                        if (rc == RC.ROW)
                        {
                            colsValues = new string[cols];
                            for (i = 0; i < cols; i++)
                            {
                                colsValues[i] = Vdbe.Column_Text(stmt, i);
                                if (colsValues[i] == null && Vdbe.Column_Type(stmt, i) != TYPE.NULL)
                                {
                                    ctx.MallocFailed = true;
                                    goto exec_out;
                                }
                            }
                        }
                        if (callback(arg, cols, colsValues, colsNames))
                        {
                            rc = RC.ABORT;
                            stmt.Finalize();
                            stmt = null;
                            Error(ctx, RC.ABORT, null);
                            goto exec_out;
                        }
                    }

                    if (rc != RC.ROW)
                    {
                        rc = stmt.Finalize();
                        stmt = null;
                        if (rc != RC.SCHEMA)
                        {
                            retrys = 0;
                            if ((sql = leftover) != string.Empty)
                            {
                                int idx = 0;
                                while (idx < sql.Length && char.IsWhiteSpace(sql[idx])) idx++;
                                if (idx != 0) sql = (idx < sql.Length ? sql.Substring(idx) : string.Empty);
                            }
                        }
                        break;
                    }
                }

                C._tagfree(ctx, ref colsNames);
                colsNames = null;
            }

        exec_out:
            if (stmt != null) stmt.Finalize();
            C._tagfree(ctx, ref colsNames);

            rc = ApiExit(ctx, rc);
            if (rc != RC.OK && C._ALWAYS(rc == ErrCode(ctx)) && errmsg != null)
                errmsg = ErrMsg(ctx);
            else if (errmsg != null)
                errmsg = null;

            Debug.Assert((rc & (RC)ctx.ErrMask) == rc);
            MutexEx.Leave(ctx.Mutex);
            return rc;
        }
    }
}
