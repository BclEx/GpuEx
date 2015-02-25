#region OMIT_COMPILEOPTION_DIAGS
#if !OMIT_COMPILEOPTION_DIAGS
using System;

namespace Core
{
    public partial class CompileTime
    {
        static string[] _compileOpt = {
#if _32BIT_ROWID
    "32BIT_ROWID",
#endif
#if _4_BYTE_ALIGNED_MALLOC
"4_BYTE_ALIGNED_MALLOC",
#endif
#if CASE_SENSITIVE_LIKE
"CASE_SENSITIVE_LIKE",
#endif
#if CHECK_PAGES
"CHECK_PAGES",
#endif
#if COVERAGE_TEST
"COVERAGE_TEST",
#endif
#if DEBUG
"DEBUG",
#endif
#if DEFAULT_LOCKING_MODE
"DEFAULT_LOCKING_MODE=" CTIMEOPT_VAL(DEFAULT_LOCKING_MODE),
#endif
#if DISABLE_DIRSYNC
"DISABLE_DIRSYNC",
#endif
#if DISABLE_LFS
"DISABLE_LFS",
#endif
#if ENABLE_ATOMIC_WRITE
"ENABLE_ATOMIC_WRITE",
#endif
#if ENABLE_CEROD
"ENABLE_CEROD",
#endif
#if ENABLE_COLUMN_METADATA
"ENABLE_COLUMN_METADATA",
#endif
#if ENABLE_EXPENSIVE_ASSERT
"ENABLE_EXPENSIVE_ASSERT",
#endif
#if ENABLE_FTS1
"ENABLE_FTS1",
#endif
#if ENABLE_FTS2
"ENABLE_FTS2",
#endif
#if ENABLE_FTS3
"ENABLE_FTS3",
#endif
#if ENABLE_FTS3_PARENTHESIS
"ENABLE_FTS3_PARENTHESIS",
#endif
#if ENABLE_FTS4
"ENABLE_FTS4",
#endif
#if ENABLE_ICU
"ENABLE_ICU",
#endif
#if ENABLE_IOTRACE
"ENABLE_IOTRACE",
#endif
#if ENABLE_LOAD_EXTENSION
"ENABLE_LOAD_EXTENSION",
#endif
#if ENABLE_LOCKING_STYLE
"ENABLE_LOCKING_STYLE=" CTIMEOPT_VAL(ENABLE_LOCKING_STYLE),
#endif
#if ENABLE_MEMORY_MANAGEMENT
"ENABLE_MEMORY_MANAGEMENT",
#endif
#if ENABLE_MEMSYS3
"ENABLE_MEMSYS3",
#endif
#if ENABLE_MEMSYS5
"ENABLE_MEMSYS5",
#endif
#if ENABLE_OVERSIZE_CELL_CHECK
"ENABLE_OVERSIZE_CELL_CHECK",
#endif
#if ENABLE_RTREE
"ENABLE_RTREE",
#endif
#if ENABLE_STAT2
"ENABLE_STAT2",
#endif
#if ENABLE_UNLOCK_NOTIFY
"ENABLE_UNLOCK_NOTIFY",
#endif
#if ENABLE_UPDATE_DELETE_LIMIT
"ENABLE_UPDATE_DELETE_LIMIT",
#endif
#if HAS_CODEC
"HAS_CODEC",
#endif
#if HAVE_ISNAN
"HAVE_ISNAN",
#endif
#if HOMEGROWN_RECURSIVE_MUTEX
"HOMEGROWN_RECURSIVE_MUTEX",
#endif
#if IGNORE_AFP_LOCK_ERRORS
"IGNORE_AFP_LOCK_ERRORS",
#endif
#if IGNORE_FLOCK_LOCK_ERRORS
"IGNORE_FLOCK_LOCK_ERRORS",
#endif
#if INT64_TYPE
"INT64_TYPE",
#endif
#if LOCK_TRACE
"LOCK_TRACE",
#endif
#if MEMDEBUG
"MEMDEBUG",
#endif
#if MIXED_ENDIAN_64BIT_FLOAT
"MIXED_ENDIAN_64BIT_FLOAT",
#endif
#if NO_SYNC
"NO_SYNC",
#endif
#if OMIT_ALTERTABLE
"OMIT_ALTERTABLE",
#endif
#if OMIT_ANALYZE
"OMIT_ANALYZE",
#endif
#if OMIT_ATTACH
"OMIT_ATTACH",
#endif
#if OMIT_AUTHORIZATION
"OMIT_AUTHORIZATION",
#endif
#if OMIT_AUTOINCREMENT
"OMIT_AUTOINCREMENT",
#endif
#if OMIT_AUTOINIT
"OMIT_AUTOINIT",
#endif
#if OMIT_AUTOMATIC_INDEX
"OMIT_AUTOMATIC_INDEX",
#endif
#if OMIT_AUTORESET
"OMIT_AUTORESET",
#endif
#if OMIT_AUTOVACUUM
"OMIT_AUTOVACUUM",
#endif
#if OMIT_BETWEEN_OPTIMIZATION
"OMIT_BETWEEN_OPTIMIZATION",
#endif
#if OMIT_BLOB_LITERAL
"OMIT_BLOB_LITERAL",
#endif
#if OMIT_BTREECOUNT
"OMIT_BTREECOUNT",
#endif
#if OMIT_BUILTIN_TEST
"OMIT_BUILTIN_TEST",
#endif
#if OMIT_CAST
"OMIT_CAST",
#endif
#if OMIT_CHECK
"OMIT_CHECK",
#endif
/* // redundant
** #if OMIT_COMPILEOPTION_DIAGS
**   "OMIT_COMPILEOPTION_DIAGS",
** #endif
*/
#if OMIT_COMPLETE
"OMIT_COMPLETE",
#endif
#if OMIT_COMPOUND_SELECT
"OMIT_COMPOUND_SELECT",
#endif
#if OMIT_DATETIME_FUNCS
"OMIT_DATETIME_FUNCS",
#endif
#if OMIT_DECLTYPE
"OMIT_DECLTYPE",
#endif
#if OMIT_DEPRECATED
"OMIT_DEPRECATED",
#endif
#if OMIT_DISKIO
"OMIT_DISKIO",
#endif
#if OMIT_EXPLAIN
"OMIT_EXPLAIN",
#endif
#if OMIT_FLAG_PRAGMAS
"OMIT_FLAG_PRAGMAS",
#endif
#if OMIT_FLOATING_POINT
"OMIT_FLOATING_POINT",
#endif
#if OMIT_FOREIGN_KEY
"OMIT_FOREIGN_KEY",
#endif
#if OMIT_GET_TABLE
"OMIT_GET_TABLE",
#endif
#if OMIT_INCRBLOB
"OMIT_INCRBLOB",
#endif
#if OMIT_INTEGRITY_CHECK
"OMIT_INTEGRITY_CHECK",
#endif
#if OMIT_LIKE_OPTIMIZATION
"OMIT_LIKE_OPTIMIZATION",
#endif
#if OMIT_LOAD_EXTENSION
"OMIT_LOAD_EXTENSION",
#endif
#if OMIT_LOCALTIME
"OMIT_LOCALTIME",
#endif
#if OMIT_LOOKASIDE
"OMIT_LOOKASIDE",
#endif
#if OMIT_MEMORYDB
"OMIT_MEMORYDB",
#endif
#if OMIT_OR_OPTIMIZATION
"OMIT_OR_OPTIMIZATION",
#endif
#if OMIT_PAGER_PRAGMAS
"OMIT_PAGER_PRAGMAS",
#endif
#if OMIT_PRAGMA
"OMIT_PRAGMA",
#endif
#if OMIT_PROGRESS_CALLBACK
"OMIT_PROGRESS_CALLBACK",
#endif
#if OMIT_QUICKBALANCE
"OMIT_QUICKBALANCE",
#endif
#if OMIT_REINDEX
"OMIT_REINDEX",
#endif
#if OMIT_SCHEMA_PRAGMAS
"OMIT_SCHEMA_PRAGMAS",
#endif
#if OMIT_SCHEMA_VERSION_PRAGMAS
"OMIT_SCHEMA_VERSION_PRAGMAS",
#endif
#if OMIT_SHARED_CACHE
"OMIT_SHARED_CACHE",
#endif
#if OMIT_SUBQUERY
"OMIT_SUBQUERY",
#endif
#if OMIT_TCL_VARIABLE
"OMIT_TCL_VARIABLE",
#endif
#if OMIT_TEMPDB
"OMIT_TEMPDB",
#endif
#if OMIT_TRACE
"OMIT_TRACE",
#endif
#if OMIT_TRIGGER
"OMIT_TRIGGER",
#endif
#if OMIT_TRUNCATE_OPTIMIZATION
"OMIT_TRUNCATE_OPTIMIZATION",
#endif
#if OMIT_UTF16
"OMIT_UTF16",
#endif
#if OMIT_VACUUM
"OMIT_VACUUM",
#endif
#if OMIT_VIEW
"OMIT_VIEW",
#endif
#if OMIT_VIRTUALTABLE
"OMIT_VIRTUALTABLE",
#endif
#if OMIT_WAL
"OMIT_WAL",
#endif
#if OMIT_WSD
"OMIT_WSD",
#endif
#if OMIT_XFER_OPT
"OMIT_XFER_OPT",
#endif
#if PERFORMANCE_TRACE
"PERFORMANCE_TRACE",
#endif
#if PROXY_DEBUG
"PROXY_DEBUG",
#endif
#if SECURE_DELETE
"SECURE_DELETE",
#endif
#if SMALL_STACK
"SMALL_STACK",
#endif
#if SOUNDEX
"SOUNDEX",
#endif
#if TCL
"TCL",
#endif
//#if TEMP_STORE
"TEMP_STORE=1",//CTIMEOPT_VAL(TEMP_STORE),
//#endif
#if TEST
"TEST",
#endif
#if THREADSAFE
"THREADSAFE=2", // For C#, hardcode to = 2 CTIMEOPT_VAL(THREADSAFE),
#else
"THREADSAFE=0", // For C#, hardcode to = 0
#endif
#if USE_ALLOCA
"USE_ALLOCA",
#endif
#if ZERO_MALLOC
"ZERO_MALLOC"
#endif
};

        public static bool OptionUsed(string optName)
        {
            if (optName.EndsWith("=")) return false;
            int length = 0;
            if (optName.StartsWith("", StringComparison.InvariantCultureIgnoreCase)) length = 7;

            // Since ArraySize(azCompileOpt) is normally in single digits, a linear search is adequate.  No need for a binary search.
            if (!string.IsNullOrEmpty(optName))
                for (int i = 0; i < _compileOpt.Length; i++)
                {
                    int n1 = (optName.Length - length < _compileOpt[i].Length) ? optName.Length - length : _compileOpt[i].Length;
                    if (string.Compare(optName, length, _compileOpt[i], 0, n1, StringComparison.InvariantCultureIgnoreCase) == 0)
                        return true;
                }
            return false;
        }

        public static string Get(int id)
        {
            return (id >= 0 && id < _compileOpt.Length ? _compileOpt[id] : null);
        }
    }
}

#endif
#endregion
