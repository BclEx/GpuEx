// table.c
#region OMIT_GET_TABLE
#if !OMIT_GET_TABLE
using System;
using System.Diagnostics;

namespace Core
{
    class TabResult
    {
        public string[] Results;
        public int ResultsLength;
        public string ErrMsg;
        public int ResultsAlloc;
        public int Rows;
        public int Columns;
        public RC RC;
    };

    public partial class Table
    {
        public static bool GetTableCallback(object arg, int columns, string[] argv, string[] colv)
        {
            TabResult p = (TabResult)arg;
            // Make sure there is enough space in p.azResult to hold everything we need to remember from this invocation of the callback.
            int need = (p.Rows == 0 && argv != null ? (int)columns * 2 : (int)columns);
            if (p.ResultsLength + need >= p.ResultsAlloc)
            {
                p.ResultsAlloc = p.ResultsAlloc * 2 + need;
                string[] newResults = new string[p.ResultsAlloc];
                if (newResults == null) goto malloc_failed;
                p.Results = newResults;
            }

            // If this is the first row, then generate an extra row containing the names of all columns.
            string z; // A single column of result
            if (p.Rows == 0)
            {
                p.Columns = (int)columns;
                for (int i = 0; i < columns; i++)
                {
                    z = colv[i];
                    if (z == null) goto malloc_failed;
                    p.Results[p.ResultsLength++] = z;
                }
            }
            else if (p.Columns != columns)
            {
                p.ErrMsg = "sqlite3_get_table() called with two or more incompatible queries";
                p.RC = RC.ERROR;
                return true;
            }

            // Copy over the row data
            if (argv != null)
            {
                for (int i = 0; i < columns; i++)
                {
                    if (argv[i] == null)
                        z = null;
                    else
                        z = argv[i];
                    p.Results[p.ResultsLength++] = z;
                }
                p.Rows++;
            }
            return false;

        malloc_failed:
            p.RC = RC.NOMEM;
            return true;
        }

        public static RC GetTable(Context db, string sql, ref string[] results, ref int rows, ref int columns, ref string errMsg)
        {
            results = null;
            columns = 0;
            rows = 0;
            errMsg = null;
            TabResult r = new TabResult();
            r.ErrMsg = null;
            r.Rows = 0;
            r.Columns = 0;
            r.RC = RC.OK;
            r.ResultsAlloc = 20;
            r.Results = new string[r.ResultsAlloc];
            r.ResultsLength = 1;
            if (r.Results == null)
                return (db.ErrCode = RC.NOMEM);
            r.Results[0] = null;
            RC rc = Main.Exec(db, sql, GetTableCallback, r, ref errMsg);
            //Debug.Assert(sizeof(r.Results[0])>= sizeof(r.ResultsLength));
            //r.Results = INT_TO_PTR(r.ResultsLength);
            if (((int)rc & 0xff) == (int)RC.ABORT)
            {
                FreeTable(ref r.Results);
                if (r.ErrMsg != null)
                {
                    if (errMsg != null)
                    {
                        C._free(ref errMsg);
                        errMsg = r.ErrMsg;
                    }
                    C._free(ref r.ErrMsg);
                }
                return (db.ErrCode = r.RC); // Assume 32-bit assignment is atomic
            }
            C._free(ref r.ErrMsg);
            if (rc != RC.OK)
            {
                FreeTable(ref r.Results);
                return rc;
            }
            if (r.ResultsAlloc > r.ResultsLength)
                Array.Resize(ref r.Results, r.ResultsLength);
            results = r.Results;
            columns = r.Columns;
            rows = r.Rows;
            return rc;
        }

        static void FreeTable(ref string[] results)
        {
        }
    }
}
#endif
#endregion

