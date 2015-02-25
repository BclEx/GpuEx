using System.Diagnostics;
namespace Core
{
    public class StatusEx
    {
        public enum STATUS
        {
            MEMORY_USED = 0,
            PAGECACHE_USED = 1,
            PAGECACHE_OVERFLOW = 2,
            SCRATCH_USED = 3,
            SCRATCH_OVERFLOW = 4,
            MALLOC_SIZE = 5,
            PARSER_STACK = 6,
            PAGECACHE_SIZE = 7,
            SCRATCH_SIZE = 8,
            MALLOC_COUNT = 9,
        }

        public class StatType
        {
            public int[] NowValue = new int[10];        // Current value
            public int[] MaxValue = new int[10];         // Maximum value
        }

        public static StatType _stat = new StatType();

        internal static int StatusValue(STATUS op)
        {
            Debug.Assert(op >= 0 && (int)op < _stat.NowValue.Length);
            return _stat.NowValue[(int)op];
        }

        internal static void StatusAdd(STATUS op, int n)
        {
            Debug.Assert(op >= 0 && (int)op < _stat.NowValue.Length);
            _stat.NowValue[(int)op] += n;
            if (_stat.NowValue[(int)op] > _stat.MaxValue[(int)op])
                _stat.MaxValue[(int)op] = _stat.NowValue[(int)op];
        }

        internal static void StatusSet(STATUS op, int x)
        {
            Debug.Assert(op >= 0 && (int)op < _stat.NowValue.Length);
            _stat.NowValue[(int)op] = x;
            if (_stat.NowValue[(int)op] > _stat.MaxValue[(int)op])
                _stat.MaxValue[(int)op] = _stat.NowValue[(int)op];
        }

        public static RC Status(STATUS op, ref int current, ref int highwater, int resetFlag)
        {
            if (op < 0 || (int)op >= _stat.NowValue.Length)
                return SysEx.MISUSE_BKPT();
            current = _stat.NowValue[(int)op];
            highwater = _stat.MaxValue[(int)op];
            if (resetFlag != 0)
                _stat.MaxValue[(int)op] = _stat.NowValue[(int)op];
            return RC.OK;
        }

        //public static SQLITE sqlite3_db_status(sqlite3 db, int op, ref int pCurrent, ref int pHighwater, int resetFlag)
        //{
        //    var rc = SQLITE.OK;
        //    MutexEx.sqlite3_mutex_enter(db.mutex);
        //    switch (op)
        //    {
        //        case SQLITE_DBSTATUS_LOOKASIDE_USED:
        //            {
        //                pCurrent = db.lookaside.nOut;
        //                pHighwater = db.lookaside.mxOut;
        //                if (resetFlag != 0)
        //                    db.lookaside.mxOut = db.lookaside.nOut;
        //                break;
        //            }
        //        case SQLITE_DBSTATUS_LOOKASIDE_HIT:
        //        case SQLITE_DBSTATUS_LOOKASIDE_MISS_SIZE:
        //        case SQLITE_DBSTATUS_LOOKASIDE_MISS_FULL:
        //            {
        //                Debug.Assert((op - SQLITE_DBSTATUS_LOOKASIDE_HIT) >= 0);
        //                Debug.Assert((op - SQLITE_DBSTATUS_LOOKASIDE_HIT) < 3);
        //                pCurrent = 0;
        //                pHighwater = db.lookaside.anStat[op - SQLITE_DBSTATUS_LOOKASIDE_HIT];
        //                if (resetFlag != 0)
        //                    db.lookaside.anStat[op - SQLITE_DBSTATUS_LOOKASIDE_HIT] = 0;
        //                break;
        //            }
        //        // Return an approximation for the amount of memory currently used by all pagers associated with the given database connection.  The
        //        // highwater mark is meaningless and is returned as zero.
        //        case SQLITE_DBSTATUS_CACHE_USED:
        //            {
        //                int totalUsed = 0;
        //                sqlite3BtreeEnterAll(db);
        //                for (var i = 0; i < db.nDb; i++)
        //                {
        //                    var pBt = db.aDb[i].pBt;
        //                    if (pBt != null)
        //                    {
        //                        Pager pPager = sqlite3BtreePager(pBt);
        //                        totalUsed += sqlite3PagerMemUsed(pPager);
        //                    }
        //                }
        //                sqlite3BtreeLeaveAll(db);
        //                pCurrent = totalUsed;
        //                pHighwater = 0;
        //                break;
        //            }
        //        // *pCurrent gets an accurate estimate of the amount of memory used to store the schema for all databases (main, temp, and any ATTACHed
        //        // databases.  *pHighwater is set to zero.
        //        case SQLITE_DBSTATUS_SCHEMA_USED:
        //            {
        //                int nByte = 0;              // Used to accumulate return value
        //                sqlite3BtreeEnterAll(db);
        //                for (var i = 0; i < db.nDb; i++)
        //                {
        //                    var pSchema = db.aDb[i].pSchema;
        //                    if (Check.ALWAYS(pSchema != null))
        //                    {
        //                        HashElem p;
        //                        for (p = sqliteHashFirst(pSchema.trigHash); p != null; p = sqliteHashNext(p))
        //                        {
        //                            var t = (ITrigger)sqliteHashData(p);
        //                            sqlite3DeleteTrigger(db, ref t);
        //                        }
        //                        for (p = sqliteHashFirst(pSchema.tblHash); p != null; p = sqliteHashNext(p))
        //                        {
        //                            var t = (ITable)sqliteHashData(p);
        //                            sqlite3DeleteTable(db, ref t);
        //                        }
        //                    }
        //                }
        //                db.pnBytesFreed = 0;
        //                sqlite3BtreeLeaveAll(db);
        //                pHighwater = 0;
        //                pCurrent = nByte;
        //                break;
        //            }
        //        // *pCurrent gets an accurate estimate of the amount of memory used to store all prepared statements.
        //        // *pHighwater is set to zero.
        //        case SQLITE_DBSTATUS_STMT_USED:
        //            {
        //                int nByte = 0;              // Used to accumulate return value
        //                for (var pVdbe = db.pVdbe; pVdbe != null; pVdbe = pVdbe.pNext)
        //                    sqlite3VdbeDeleteObject(db, ref pVdbe);
        //                db.pnBytesFreed = 0;
        //                pHighwater = 0;
        //                pCurrent = nByte;
        //                break;
        //            }

        //        default:
        //            {
        //                rc = RC.ERROR;
        //                break;
        //            }
        //    }
        //    MutexEx.sqlite3_mutex_leave(db.mutex);
        //    return rc;
        //}
    }
}
