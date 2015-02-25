#region OMIT_VACUUM
#if !OMIT_VACUUM && !OMIT_ATTACH
using System;
using System.Diagnostics;

namespace Core.Command
{
    public class Vacuum
    {
        static RC VacuumFinalize(Context ctx, Vdbe stmt, string errMsg)
        {
            RC rc = sqlite3VdbeFinalize(ref stmt);
            if (rc != RC.OK)
                C._setstring(ref errMsg, ctx, sqlite3_errmsg(ctx));
            return rc;
        }

        static RC ExecSql(Context ctx, string errMsg, string sql)
        {
            if (sql == null) return RC.NOMEM;
            Vdbe stmt = null;
            if (sqlite3_prepare(ctx, sql, -1, ref stmt, 0) != RC.OK)
            {
                C._setstring(ref errMsg, ctx, sqlite3_errmsg(ctx));
                return sqlite3_errcode(ctx);
            }
#if !NDEBUG
            RC rc =
#endif
 sqlite3_step(stmt);
            Debug.Assert(rc != RC.ROW || (ctx.Flags & Context.FLAG.CountRows) != 0);
            return VacuumFinalize(ctx, stmt, errMsg);
        }

        static RC ExecExecSql(Context ctx, string errMsg, string sql)
        {
            Vdbe stmt = null;
            RC rc = sqlite3_prepare(ctx, sql, -1, ref stmt, 0);
            if (rc != RC.OK) return rc;

            while (sqlite3_step(stmt) == RC.ROW)
            {
                rc = ExecSql(ctx, errMsg, sqlite3_column_text(stmt, 0));
                if (rc != RC.OK)
                {
                    VacuumFinalize(ctx, stmt, errMsg);
                    return rc;
                }
            }
            return VacuumFinalize(ctx, stmt, errMsg);
        }

        public static void Vacuum(Parse parse)
        {
            Vdbe v = parse.GetVdbe();
            if (v != null)
                v.AddOp2(OP.Vacuum, 0, 0);
            return;
        }

        byte[] _runVacuum_copy = new byte[]  {
            BTREE_SCHEMA_VERSION,     1,  // Add one to the old schema cookie
            BTREE_DEFAULT_CACHE_SIZE, 0,  // Preserve the default page cache size
            BTREE_TEXT_ENCODING,      0,  // Preserve the text encoding
            BTREE_USER_VERSION,       0,  // Preserve the user version
        };
        public static RC RunVacuum(ref string errMsg, Context ctx)
        {

            if (ctx.AutoCommit == 0)
            {
                C._setstring(ref errMsg, ctx, "cannot VACUUM from within a transaction");
                return RC.ERROR;
            }
            if (ctx.ActiveVdbeCnt > 1)
            {
                C._setstring(ref errMsg, ctx, "cannot VACUUM - SQL statements in progress");
                return RC.ERROR;
            }

            // Save the current value of the database flags so that it can be restored before returning. Then set the writable-schema flag, and
            // disable CHECK and foreign key constraints.
            int saved_Flags = ctx.Flags; // Saved value of the db.flags
            int saved_Changes = ctx.Changes; // Saved value of db.nChange
            int saved_TotalChanges = ctx.TotalChanges; // Saved value of db.nTotalChange
            Action<object, string> saved_Trace = ctx.Trace; // Saved db->xTrace
            ctx.Flags |= Context.FLAG.WriteSchema | Context.FLAG.IgnoreChecks | Context.FLAG.PreferBuiltin;
            ctx.Flags &= ~(Context.FLAG.ForeignKeys | Context.FLAG.ReverseOrder);
            ctx.Trace = null;

            Btree main = ctx.DBs[0].Bt; // The database being vacuumed
            bool isMemDb = sqlite3PagerIsMemdb(main.get_Pager()); // True if vacuuming a :memory: database

            // Attach the temporary database as 'vacuum_db'. The synchronous pragma can be set to 'off' for this file, as it is not recovered if a crash
            // occurs anyway. The integrity of the database is maintained by a (possibly synchronous) transaction opened on the main database before
            // sqlite3BtreeCopyFile() is called.
            //
            // An optimisation would be to use a non-journaled pager. (Later:) I tried setting "PRAGMA vacuum_db.journal_mode=OFF" but
            // that actually made the VACUUM run slower.  Very little journalling actually occurs when doing a vacuum since the vacuum_db is initially
            // empty.  Only the journal header is written.  Apparently it takes more time to parse and run the PRAGMA to turn journalling off than it does
            // to write the journal header file.
            int db = ctx.DBs.length; // Number of attached databases
            string sql = (sqlite3TempInMemory(ctx) ? "ATTACH ':memory:' AS vacuum_db;" : "ATTACH '' AS vacuum_db;"); // SQL statements
            RC rc = ExecSql(ctx, errMsg, sql); // Return code from service routines
            Context.DB dbObj = null; // Database to detach at end of vacuum
            if (ctx.DBs.length > db)
            {
                dbObj = ctx.DBs[ctx.DBs.length - 1];
                Debug.Assert(dbObj.Name == "vacuum_db");
            }
            if (rc != RC.OK) goto end_of_vacuum;
            Btree temp = ctx.DBs[ctx.DBs.length - 1].Bt; // The temporary database we vacuum into

            // The call to execSql() to attach the temp database has left the file locked (as there was more than one active statement when the transaction
            // to read the schema was concluded. Unlock it here so that this doesn't cause problems for the call to BtreeSetPageSize() below.
            temp.Commit();

            int res = main.GetReserve(); // Bytes of reserved space at the end of each page

            // A VACUUM cannot change the pagesize of an encrypted database.
#if HAS_CODEC
            if (ctx.NextPagesize != 0)
            {
                int nKey;
                string zKey;
                sqlite3CodecGetKey(ctx, 0, out zKey, out nKey);
                if (nKey != 0) ctx.NextPagesize = 0;
            }
#endif
            rc = ExecSql(ctx, errMsg, "PRAGMA vacuum_db.synchronous=OFF");
            if (rc != RC.OK) goto end_of_vacuum;

            // Begin a transaction and take an exclusive lock on the main database file. This is done before the sqlite3BtreeGetPageSize(main) call below,
            // to ensure that we do not try to change the page-size on a WAL database.
            rc = ExecSql(ctx, errMsg, "BEGIN;");
            if (rc != RC.OK) goto end_of_vacuum;
            rc = main.BeginTrans(2);
            if (rc != RC.OK) goto end_of_vacuum;

            // Do not attempt to change the page size for a WAL database
            if (main.get_Pager().GetJournalMode() == IPager.JOURNALMODE.WAL)
                ctx.NextPagesize = 0;

            if (temp.SetPageSize(main.GetPageSize(), res, 0) != 0 || (!isMemDb && temp.SetPageSize(ctx->NextPagesize, res, false) != 0) || C._NEVER(ctx.MallocFailed))
            {
                rc = RC.NOMEM;
                goto end_of_vacuum;
            }

#if !OMIT_AUTOVACUUM
            temp.SetAutoVacuum(ctx.NextAutovac >= 0 ? ctx.NextAutovac : main.GetAutoVacuum());
#endif

            // Query the schema of the main database. Create a mirror schema in the temporary database.
            rc = ExecExecSql(ctx, errMsg,
                "SELECT 'CREATE TABLE vacuum_db.' || substr(sql,14) " +
                "  FROM sqlite_master WHERE type='table' AND name!='sqlite_sequence'" +
                "   AND rootpage>0"
            );
            if (rc != RC.OK) goto end_of_vacuum;
            rc = ExecExecSql(ctx, errMsg,
                "SELECT 'CREATE INDEX vacuum_db.' || substr(sql,14)" +
                "  FROM sqlite_master WHERE sql LIKE 'CREATE INDEX %' ");
            if (rc != RC.OK) goto end_of_vacuum;
            rc = ExecExecSql(ctx, errMsg,
                "SELECT 'CREATE UNIQUE INDEX vacuum_db.' || substr(sql,21) " +
                "  FROM sqlite_master WHERE sql LIKE 'CREATE UNIQUE INDEX %'");
            if (rc != RC.OK) goto end_of_vacuum;

            // Loop through the tables in the main database. For each, do an "INSERT INTO vacuum_db.xxx SELECT * FROM main.xxx;" to copy
            // the contents to the temporary database.
            rc = ExecExecSql(ctx, errMsg,
                "SELECT 'INSERT INTO vacuum_db.' || quote(name) " +
                "|| ' SELECT * FROM main.' || quote(name) || ';'" +
                "FROM main.sqlite_master " +
                "WHERE type = 'table' AND name!='sqlite_sequence' " +
                "  AND rootpage>0");
            if (rc != RC.OK) goto end_of_vacuum;

            // Copy over the sequence table
            rc = ExecExecSql(ctx, errMsg,
                "SELECT 'DELETE FROM vacuum_db.' || quote(name) || ';' " +
                "FROM vacuum_db.sqlite_master WHERE name='sqlite_sequence' ");
            if (rc != RC.OK) goto end_of_vacuum;
            rc = ExecExecSql(ctx, errMsg,
                "SELECT 'INSERT INTO vacuum_db.' || quote(name) " +
                "|| ' SELECT * FROM main.' || quote(name) || ';' " +
                "FROM vacuum_db.sqlite_master WHERE name=='sqlite_sequence';");
            if (rc != RC.OK) goto end_of_vacuum;

            // Copy the triggers, views, and virtual tables from the main database over to the temporary database.  None of these objects has any
            // associated storage, so all we have to do is copy their entries from the SQLITE_MASTER table.
            rc = ExecSql(ctx, errMsg,
                "INSERT INTO vacuum_db.sqlite_master " +
                "  SELECT type, name, tbl_name, rootpage, sql" +
                "    FROM main.sqlite_master" +
                "   WHERE type='view' OR type='trigger'" +
                "      OR (type='table' AND rootpage=0)");
            if (rc != 0) goto end_of_vacuum;

            // At this point, there is a write transaction open on both the vacuum database and the main database. Assuming no error occurs,
            // both transactions are closed by this block - the main database transaction by sqlite3BtreeCopyFile() and the other by an explicit
            // call to sqlite3BtreeCommit().
            {

                // This array determines which meta meta values are preserved in the vacuum.  Even entries are the meta value number and odd entries
                // are an increment to apply to the meta value after the vacuum. The increment is used to increase the schema cookie so that other
                // connections to the same database will know to reread the schema.
                Debug.Assert(Btree.IsInTrans(temp));
                Debug.Assert(Btree.IsInTrans(main));

                // Copy Btree meta values
                for (int i = 0; i < _runVacuum_copy.Length; i += 2)
                {
                    // GetMeta() and UpdateMeta() cannot fail in this context because we already have page 1 loaded into cache and marked dirty.
                    uint meta = 0;
                    main.GetMeta(_runVacuum_copy[i], ref meta);
                    rc = temp.UpdateMeta(temp, _runVacuum_copy[i], (uint)(meta + _runVacuum_copy[i + 1]));
                    if (C._NEVER(rc != RC.OK)) goto end_of_vacuum;
                }

                rc = sqlite3BtreeCopyFile(main, temp);
                if (rc != RC.OK) goto end_of_vacuum;
                rc = temp.Commit();
                if (rc != RC.OK) goto end_of_vacuum;
#if !SQLITE_OMIT_AUTOVACUUM
                main.SetAutoVacuum(temp.GetAutoVacuum());
#endif
            }

            Debug.Assert(rc == RC.OK);
            rc = main.SetPageSize(temp.GetPageSize(temp), res, 1);

        end_of_vacuum:
            // Restore the original value of ctx->flags
            ctx.Flags = saved_Flags;
            ctx.Changes = saved_Changes;
            ctx.TotalChanges = saved_TotalChanges;
            ctx.Trace = saved_Trace;
            sqlite3BtreeSetPageSize(main, -1, -1, 1);

            // Currently there is an SQL level transaction open on the vacuum database. No locks are held on any other files (since the main file
            // was committed at the btree level). So it safe to end the transaction by manually setting the autoCommit flag to true and detaching the
            // vacuum database. The vacuum_db journal file is deleted when the pager is closed by the DETACH.
            ctx.AutoCommit = 1;

            if (dbObj != null)
            {
                dbObj.Bt.Close();
                dbObj.Bt = null;
                dbObj.Schema = null;
            }

            // This both clears the schemas and reduces the size of the db->aDb[] array.
            sqlite3ResetInternalSchema(ctx, -1);

            return rc;
        }
    }
}
#endif
#endregion