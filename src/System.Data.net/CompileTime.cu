#pragma region OMIT_COMPILEOPTION_DIAGS
#ifndef OMIT_COMPILEOPTION_DIAGS
#include <Core/Core.cu.h>

namespace Core
{
	// These macros are provided to "stringify" the value of the define for those options in which the value is meaningful.
#define CTIMEOPT_VAL_(opt) #opt
#define CTIMEOPT_VAL(opt) CTIMEOPT_VAL_(opt)

	__constant__ static const char *const _compileOpt[] = {
#ifdef _32BIT_ROWID
		"32BIT_ROWID",
#endif
#ifdef _4_BYTE_ALIGNED_MALLOC
		"4_BYTE_ALIGNED_MALLOC",
#endif
#ifdef CASE_SENSITIVE_LIKE
		"CASE_SENSITIVE_LIKE",
#endif
#ifdef CHECK_PAGES
		"CHECK_PAGES",
#endif
#ifdef COVERAGE_TEST
		"COVERAGE_TEST",
#endif
#ifdef CURDIR
		"CURDIR",
#endif
#ifdef _DEBUG
		"DEBUG",
#endif
#ifdef DEFAULT_LOCKING_MODE
		"DEFAULT_LOCKING_MODE=" CTIMEOPT_VAL(DEFAULT_LOCKING_MODE),
#endif
#ifdef DISABLE_DIRSYNC
		"DISABLE_DIRSYNC",
#endif
#ifdef DISABLE_LFS
		"DISABLE_LFS",
#endif
#ifdef ENABLE_ATOMIC_WRITE
		"ENABLE_ATOMIC_WRITE",
#endif
#ifdef ENABLE_CEROD
		"ENABLE_CEROD",
#endif
#ifdef ENABLE_COLUMN_METADATA
		"ENABLE_COLUMN_METADATA",
#endif
#ifdef ENABLE_EXPENSIVE_ASSERT
		"ENABLE_EXPENSIVE_ASSERT",
#endif
#ifdef ENABLE_FTS1
		"ENABLE_FTS1",
#endif
#ifdef ENABLE_FTS2
		"ENABLE_FTS2",
#endif
#ifdef ENABLE_FTS3
		"ENABLE_FTS3",
#endif
#ifdef ENABLE_FTS3_PARENTHESIS
		"ENABLE_FTS3_PARENTHESIS",
#endif
#ifdef ENABLE_FTS4
		"ENABLE_FTS4",
#endif
#ifdef ENABLE_ICU
		"ENABLE_ICU",
#endif
#ifdef ENABLE_IOTRACE
		"ENABLE_IOTRACE",
#endif
#ifdef ENABLE_LOAD_EXTENSION
		"ENABLE_LOAD_EXTENSION",
#endif
#ifdef ENABLE_LOCKING_STYLE
		"ENABLE_LOCKING_STYLE=" CTIMEOPT_VAL(ENABLE_LOCKING_STYLE),
#endif
#ifdef ENABLE_MEMORY_MANAGEMENT
		"ENABLE_MEMORY_MANAGEMENT",
#endif
#ifdef ENABLE_MEMSYS3
		"ENABLE_MEMSYS3",
#endif
#ifdef ENABLE_MEMSYS5
		"ENABLE_MEMSYS5",
#endif
#ifdef ENABLE_OVERSIZE_CELL_CHECK
		"ENABLE_OVERSIZE_CELL_CHECK",
#endif
#ifdef ENABLE_RTREE
		"ENABLE_RTREE",
#endif
#ifdef ENABLE_STAT3
		"ENABLE_STAT3",
#endif
#ifdef ENABLE_UNLOCK_NOTIFY
		"ENABLE_UNLOCK_NOTIFY",
#endif
#ifdef ENABLE_UPDATE_DELETE_LIMIT
		"ENABLE_UPDATE_DELETE_LIMIT",
#endif
#ifdef HAS_CODEC
		"HAS_CODEC",
#endif
#ifdef HAVE_ISNAN
		"HAVE_ISNAN",
#endif
#ifdef HOMEGROWN_RECURSIVE_MUTEX
		"HOMEGROWN_RECURSIVE_MUTEX",
#endif
#ifdef IGNORE_AFP_LOCK_ERRORS
		"IGNORE_AFP_LOCK_ERRORS",
#endif
#ifdef IGNORE_FLOCK_LOCK_ERRORS
		"IGNORE_FLOCK_LOCK_ERRORS",
#endif
#ifdef INT64_TYPE
		"INT64_TYPE",
#endif
#ifdef LOCK_TRACE
		"LOCK_TRACE",
#endif
#ifdef MAX_SCHEMA_RETRY
		"MAX_SCHEMA_RETRY=" CTIMEOPT_VAL(MAX_SCHEMA_RETRY),
#endif
#ifdef MEMDEBUG
		"MEMDEBUG",
#endif
#ifdef MIXED_ENDIAN_64BIT_FLOAT
		"MIXED_ENDIAN_64BIT_FLOAT",
#endif
#ifdef NO_SYNC
		"NO_SYNC",
#endif
#ifdef OMIT_ALTERTABLE
		"OMIT_ALTERTABLE",
#endif
#ifdef OMIT_ANALYZE
		"OMIT_ANALYZE",
#endif
#ifdef OMIT_ATTACH
		"OMIT_ATTACH",
#endif
#ifdef OMIT_AUTHORIZATION
		"OMIT_AUTHORIZATION",
#endif
#ifdef OMIT_AUTOINCREMENT
		"OMIT_AUTOINCREMENT",
#endif
#ifdef OMIT_AUTOINIT
		"OMIT_AUTOINIT",
#endif
#ifdef OMIT_AUTOMATIC_INDEX
		"OMIT_AUTOMATIC_INDEX",
#endif
#ifdef OMIT_AUTORESET
		"OMIT_AUTORESET",
#endif
#ifdef OMIT_AUTOVACUUM
		"OMIT_AUTOVACUUM",
#endif
#ifdef OMIT_BETWEEN_OPTIMIZATION
		"OMIT_BETWEEN_OPTIMIZATION",
#endif
#ifdef OMIT_BLOB_LITERAL
		"OMIT_BLOB_LITERAL",
#endif
#ifdef OMIT_BTREECOUNT
		"OMIT_BTREECOUNT",
#endif
#ifdef OMIT_BUILTIN_TEST
		"OMIT_BUILTIN_TEST",
#endif
#ifdef OMIT_CAST
		"OMIT_CAST",
#endif
#ifdef OMIT_CHECK
		"OMIT_CHECK",
#endif
		// redundant
		//#ifdef OMIT_COMPILEOPTION_DIAGS
		//   "OMIT_COMPILEOPTION_DIAGS",
		//#endif
#ifdef OMIT_COMPLETE
		"OMIT_COMPLETE",
#endif
#ifdef OMIT_COMPOUND_SELECT
		"OMIT_COMPOUND_SELECT",
#endif
#ifdef OMIT_DATETIME_FUNCS
		"OMIT_DATETIME_FUNCS",
#endif
#ifdef OMIT_DECLTYPE
		"OMIT_DECLTYPE",
#endif
#ifdef OMIT_DEPRECATED
		"OMIT_DEPRECATED",
#endif
#ifdef OMIT_DISKIO
		"OMIT_DISKIO",
#endif
#ifdef OMIT_EXPLAIN
		"OMIT_EXPLAIN",
#endif
#ifdef OMIT_FLAG_PRAGMAS
		"OMIT_FLAG_PRAGMAS",
#endif
#ifdef OMIT_FLOATING_POINT
		"OMIT_FLOATING_POINT",
#endif
#ifdef OMIT_FOREIGN_KEY
		"OMIT_FOREIGN_KEY",
#endif
#ifdef OMIT_GET_TABLE
		"OMIT_GET_TABLE",
#endif
#ifdef OMIT_INCRBLOB
		"OMIT_INCRBLOB",
#endif
#ifdef OMIT_INTEGRITY_CHECK
		"OMIT_INTEGRITY_CHECK",
#endif
#ifdef OMIT_LIKE_OPTIMIZATION
		"OMIT_LIKE_OPTIMIZATION",
#endif
#ifdef OMIT_LOAD_EXTENSION
		"OMIT_LOAD_EXTENSION",
#endif
#ifdef OMIT_LOCALTIME
		"OMIT_LOCALTIME",
#endif
#ifdef OMIT_LOOKASIDE
		"OMIT_LOOKASIDE",
#endif
#ifdef OMIT_MEMORYDB
		"OMIT_MEMORYDB",
#endif
#ifdef OMIT_OR_OPTIMIZATION
		"OMIT_OR_OPTIMIZATION",
#endif
#ifdef OMIT_PAGER_PRAGMAS
		"OMIT_PAGER_PRAGMAS",
#endif
#ifdef OMIT_PRAGMA
		"OMIT_PRAGMA",
#endif
#ifdef OMIT_PROGRESS_CALLBACK
		"OMIT_PROGRESS_CALLBACK",
#endif
#ifdef OMIT_QUICKBALANCE
		"OMIT_QUICKBALANCE",
#endif
#ifdef OMIT_REINDEX
		"OMIT_REINDEX",
#endif
#ifdef OMIT_SCHEMA_PRAGMAS
		"OMIT_SCHEMA_PRAGMAS",
#endif
#ifdef OMIT_SCHEMA_VERSION_PRAGMAS
		"OMIT_SCHEMA_VERSION_PRAGMAS",
#endif
#ifdef OMIT_SHARED_CACHE
		"OMIT_SHARED_CACHE",
#endif
#ifdef OMIT_SUBQUERY
		"OMIT_SUBQUERY",
#endif
#ifdef OMIT_TCL_VARIABLE
		"OMIT_TCL_VARIABLE",
#endif
#ifdef OMIT_TEMPDB
		"OMIT_TEMPDB",
#endif
#ifdef OMIT_TRACE
		"OMIT_TRACE",
#endif
#ifdef OMIT_TRIGGER
		"OMIT_TRIGGER",
#endif
#ifdef OMIT_TRUNCATE_OPTIMIZATION
		"OMIT_TRUNCATE_OPTIMIZATION",
#endif
#ifdef OMIT_UTF16
		"OMIT_UTF16",
#endif
#ifdef OMIT_VACUUM
		"OMIT_VACUUM",
#endif
#ifdef OMIT_VIEW
		"OMIT_VIEW",
#endif
#ifdef OMIT_VIRTUALTABLE
		"OMIT_VIRTUALTABLE",
#endif
#ifdef OMIT_WAL
		"OMIT_WAL",
#endif
#ifdef OMIT_WSD
		"OMIT_WSD",
#endif
#ifdef OMIT_XFER_OPT
		"OMIT_XFER_OPT",
#endif
#ifdef PERFORMANCE_TRACE
		"PERFORMANCE_TRACE",
#endif
#ifdef PROXY_DEBUG
		"PROXY_DEBUG",
#endif
#ifdef RTREE_INT_ONLY
		"RTREE_INT_ONLY",
#endif
#ifdef SECURE_DELETE
		"SECURE_DELETE",
#endif
#ifdef SMALL_STACK
		"SMALL_STACK",
#endif
#ifdef SOUNDEX
		"SOUNDEX",
#endif
#ifdef TCL
		"TCL",
#endif
#ifdef TEMP_STORE
		"TEMP_STORE=" CTIMEOPT_VAL(TEMP_STORE),
#endif
#ifdef TEST
		"TEST",
#endif
#ifdef THREADSAFE
		"THREADSAFE=" CTIMEOPT_VAL(THREADSAFE),
#endif
#ifdef USE_ALLOCA
		"USE_ALLOCA",
#endif
#ifdef ZERO_MALLOC
		"ZERO_MALLOC"
#endif
	};

	__device__ bool CompileTimeOptionUsed(const char *optName)
	{
		if (!_strncmp(optName, "", 7)) optName += 7;
		int length = _strlen30(optName);
		// Since ArraySize(azCompileOpt) is normally in single digits, a linear search is adequate.  No need for a binary search.
		for (int i = 0; i < _lengthof(_compileOpt); i++)
			if (!_strncmp(optName, _compileOpt[i], length) && (_compileOpt[i][length] == 0 || _compileOpt[i][length] == '='))
				return true;
		return false;
	}

	__device__ const char *CompileTimeGet(int id)
	{
		return (id >= 0 && id < _lengthof(_compileOpt) ? _compileOpt[id] : nullptr);
	}

}
#endif
#pragma endregion