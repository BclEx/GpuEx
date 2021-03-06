// This file contains code used for testing the SQLite system. None of the code in this file goes into a deliverable build.
// 
// The focus of this file is providing the TCL testing layer access to compile-time constants.
//#include "sqliteLimit.h"
#include <RuntimeTypes.h>
#include <Core+Vdbe\VdbeInt.cu.h>
#include <Jim.h>

// Macro to stringify the results of the evaluation a pre-processor macro. i.e. so that STRINGVALUE(SQLITE_NOMEM) -> "7".
#define STRINGVALUE2(x) #x
#define STRINGVALUE(x) STRINGVALUE2(x)

// This routine sets entries in the global ::sqlite_options() array variable according to the compile-time configuration of the database.  Test
// procedures use this to determine when tests should be omitted.
__device__ static void set_options(Jim_Interp *interp)
{
#ifdef HAVE_MALLOC_USABLE_SIZE
	Jim_SetVar2(interp, "sqlite_options", "malloc_usable_size", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "malloc_usable_size", "0", JIM_GLOBAL);
#endif

#ifdef _32BIT_ROWID
	Jim_SetVar2(interp, "sqlite_options", "rowid32", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "rowid32", "0", JIM_GLOBAL);
#endif

#ifdef CASE_SENSITIVE_LIKE
	Jim_SetVar2(interp, "sqlite_options", "casesensitivelike", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "casesensitivelike", "0", JIM_GLOBAL);
#endif

#ifdef CURDIR
	Jim_SetVar2(interp, "sqlite_options", "curdir", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "curdir", "0", JIM_GLOBAL);
#endif

#ifdef _DEBUG
	Jim_SetVar2(interp, "sqlite_options", "debug", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "debug", "0", JIM_GLOBAL);
#endif

#ifdef DIRECT_OVERFLOW_READ
	Jim_SetVar2(interp, "sqlite_options", "direct_read", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "direct_read", "0", JIM_GLOBAL);
#endif

#ifdef DISABLE_DIRSYNC
	Jim_SetVar2(interp, "sqlite_options", "dirsync", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "dirsync", "1", JIM_GLOBAL);
#endif

#ifdef DISABLE_LFS
	Jim_SetVar2(interp, "sqlite_options", "lfs", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "lfs", "1", JIM_GLOBAL);
#endif

#if 1 /* def SQLITE_MEMDEBUG */
	Jim_SetVar2(interp, "sqlite_options", "memdebug", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "memdebug", "0", JIM_GLOBAL);
#endif

#ifdef ENABLE_8_3_NAMES
	Jim_SetVar2(interp, "sqlite_options", "8_3_names", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "8_3_names", "0", JIM_GLOBAL);
#endif

#ifdef ENABLE_MEMSYS3
	Jim_SetVar2(interp, "sqlite_options", "mem3", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "mem3", "0", JIM_GLOBAL);
#endif

#ifdef ENABLE_MEMSYS5
	Jim_SetVar2(interp, "sqlite_options", "mem5", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "mem5", "0", JIM_GLOBAL);
#endif

#ifdef MUTEX_OMIT
	Jim_SetVar2(interp, "sqlite_options", "mutex", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "mutex", "1", JIM_GLOBAL);
#endif

#ifdef MUTEX_NOOP
	Jim_SetVar2(interp, "sqlite_options", "mutex_noop", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "mutex_noop", "0", JIM_GLOBAL);
#endif

#ifdef OMIT_ALTERTABLE
	Jim_SetVar2(interp, "sqlite_options", "altertable", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "altertable", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_ANALYZE
	Jim_SetVar2(interp, "sqlite_options", "analyze", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "analyze", "1", JIM_GLOBAL);
#endif

#ifdef ENABLE_ATOMIC_WRITE
	Jim_SetVar2(interp, "sqlite_options", "atomicwrite", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "atomicwrite", "0", JIM_GLOBAL);
#endif

#ifdef OMIT_ATTACH
	Jim_SetVar2(interp, "sqlite_options", "attach", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "attach", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_AUTHORIZATION
	Jim_SetVar2(interp, "sqlite_options", "auth", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "auth", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_AUTOINCREMENT
	Jim_SetVar2(interp, "sqlite_options", "autoinc", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "autoinc", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_AUTOMATIC_INDEX
	Jim_SetVar2(interp, "sqlite_options", "autoindex", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "autoindex", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_AUTORESET
	Jim_SetVar2(interp, "sqlite_options", "autoreset", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "autoreset", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_AUTOVACUUM
	Jim_SetVar2(interp, "sqlite_options", "autovacuum", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "autovacuum", "1", JIM_GLOBAL);
#endif
#if !defined(DEFAULT_AUTOVACUUM)
	Jim_SetVar2(interp,"sqlite_options","default_autovacuum", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "default_autovacuum", STRINGVALUE(DEFAULT_AUTOVACUUM), JIM_GLOBAL);
#endif

#ifdef OMIT_BETWEEN_OPTIMIZATION
	Jim_SetVar2(interp, "sqlite_options", "between_opt", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "between_opt", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_BUILTIN_TEST
	Jim_SetVar2(interp, "sqlite_options", "builtin_test", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "builtin_test", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_BLOB_LITERAL
	Jim_SetVar2(interp, "sqlite_options", "bloblit", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "bloblit", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_CAST
	Jim_SetVar2(interp, "sqlite_options", "cast", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "cast", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_CHECK
	Jim_SetVar2(interp, "sqlite_options", "check", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "check", "1", JIM_GLOBAL);
#endif

#ifdef ENABLE_COLUMN_METADATA
	Jim_SetVar2(interp, "sqlite_options", "columnmetadata", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "columnmetadata", "0", JIM_GLOBAL);
#endif

#ifdef ENABLE_OVERSIZE_CELL_CHECK
	Jim_SetVar2(interp, "sqlite_options", "oversize_cell_check", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "oversize_cell_check", "0", JIM_GLOBAL);
#endif

#ifdef OMIT_COMPILEOPTION_DIAGS
	Jim_SetVar2(interp, "sqlite_options", "compileoption_diags", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "compileoption_diags", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_COMPLETE
	Jim_SetVar2(interp, "sqlite_options", "complete", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "complete", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_COMPOUND_SELECT
	Jim_SetVar2(interp, "sqlite_options", "compound", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "compound", "1", JIM_GLOBAL);
#endif

	Jim_SetVar2(interp, "sqlite_options", "conflict", "1", JIM_GLOBAL);

#if OS_UNIX
	Jim_SetVar2(interp, "sqlite_options", "crashtest", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "crashtest", "0", JIM_GLOBAL);
#endif

#ifdef OMIT_DATETIME_FUNCS
	Jim_SetVar2(interp, "sqlite_options", "datetime", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "datetime", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_DECLTYPE
	Jim_SetVar2(interp, "sqlite_options", "decltype", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "decltype", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_DEPRECATED
	Jim_SetVar2(interp, "sqlite_options", "deprecated", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "deprecated", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_DISKIO
	Jim_SetVar2(interp, "sqlite_options", "diskio", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "diskio", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_EXPLAIN
	Jim_SetVar2(interp, "sqlite_options", "explain", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "explain", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_FLOATING_POINT
	Jim_SetVar2(interp, "sqlite_options", "floatingpoint", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "floatingpoint", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_FOREIGN_KEY
	Jim_SetVar2(interp, "sqlite_options", "foreignkey", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "foreignkey", "1", JIM_GLOBAL);
#endif

#ifdef ENABLE_FTS1
	Jim_SetVar2(interp, "sqlite_options", "fts1", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "fts1", "0", JIM_GLOBAL);
#endif

#ifdef ENABLE_FTS2
	Jim_SetVar2(interp, "sqlite_options", "fts2", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "fts2", "0", JIM_GLOBAL);
#endif

#ifdef ENABLE_FTS3
	Jim_SetVar2(interp, "sqlite_options", "fts3", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "fts3", "0", JIM_GLOBAL);
#endif

#if defined(ENABLE_FTS3) && defined(ENABLE_FTS4_UNICODE61)
	Jim_SetVar2(interp, "sqlite_options", "fts3_unicode", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "fts3_unicode", "0", JIM_GLOBAL);
#endif

#ifdef DISABLE_FTS4_DEFERRED
	Jim_SetVar2(interp, "sqlite_options", "fts4_deferred", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "fts4_deferred", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_GET_TABLE
	Jim_SetVar2(interp, "sqlite_options", "gettable", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "gettable", "1", JIM_GLOBAL);
#endif

#ifdef ENABLE_ICU
	Jim_SetVar2(interp, "sqlite_options", "icu", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "icu", "0", JIM_GLOBAL);
#endif

#ifdef OMIT_INCRBLOB
	Jim_SetVar2(interp, "sqlite_options", "incrblob", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "incrblob", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_INTEGRITY_CHECK
	Jim_SetVar2(interp, "sqlite_options", "integrityck", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "integrityck", "1", JIM_GLOBAL);
#endif

#if defined(DEFAULT_FILE_FORMAT) && DEFAULT_FILE_FORMAT==1
	Jim_SetVar2(interp, "sqlite_options", "legacyformat", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "legacyformat", "0", JIM_GLOBAL);
#endif

#ifdef OMIT_LIKE_OPTIMIZATION
	Jim_SetVar2(interp, "sqlite_options", "like_opt", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "like_opt", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_LOAD_EXTENSION
	Jim_SetVar2(interp, "sqlite_options", "load_ext", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "load_ext", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_LOCALTIME
	Jim_SetVar2(interp, "sqlite_options", "localtime", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "localtime", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_LOOKASIDE
	Jim_SetVar2(interp, "sqlite_options", "lookaside", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "lookaside", "1", JIM_GLOBAL);
#endif

	Jim_SetVar2(interp, "sqlite_options", "long_double", (sizeof(double64) > sizeof(double) ? "1" : "0"), JIM_GLOBAL);

#ifdef OMIT_MEMORYDB
	Jim_SetVar2(interp, "sqlite_options", "memorydb", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "memorydb", "1", JIM_GLOBAL);
#endif

#ifdef ENABLE_MEMORY_MANAGEMENT
	Jim_SetVar2(interp, "sqlite_options", "memorymanage", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "memorymanage", "0", JIM_GLOBAL);
#endif

	Jim_SetVar2(interp, "sqlite_options", "mergesort", "1", JIM_GLOBAL);

#ifdef OMIT_OR_OPTIMIZATION
	Jim_SetVar2(interp, "sqlite_options", "or_opt", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "or_opt", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_PAGER_PRAGMAS
	Jim_SetVar2(interp, "sqlite_options", "pager_pragmas", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "pager_pragmas", "1", JIM_GLOBAL);
#endif

#if defined(OMIT_PRAGMA) || defined(OMIT_FLAG_PRAGMAS)
	Jim_SetVar2(interp, "sqlite_options", "pragma", "0", JIM_GLOBAL);
	Jim_SetVar2(interp, "sqlite_options", "integrityck", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "pragma", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_PROGRESS_CALLBACK
	Jim_SetVar2(interp, "sqlite_options", "progress", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "progress", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_REINDEX
	Jim_SetVar2(interp, "sqlite_options", "reindex", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "reindex", "1", JIM_GLOBAL);
#endif

#ifdef ENABLE_RTREE
	Jim_SetVar2(interp, "sqlite_options", "rtree", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "rtree", "0", JIM_GLOBAL);
#endif

#ifdef RTREE_INT_ONLY
	Jim_SetVar2(interp, "sqlite_options", "rtree_int_only", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "rtree_int_only", "0", JIM_GLOBAL);
#endif

#ifdef OMIT_SCHEMA_PRAGMAS
	Jim_SetVar2(interp, "sqlite_options", "schema_pragmas", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "schema_pragmas", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_SCHEMA_VERSION_PRAGMAS
	Jim_SetVar2(interp, "sqlite_options", "schema_version", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "schema_version", "1", JIM_GLOBAL);
#endif

#ifdef ENABLE_STAT3
	Jim_SetVar2(interp, "sqlite_options", "stat3", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "stat3", "0", JIM_GLOBAL);
#endif

#if !defined(ENABLE_LOCKING_STYLE)
#if defined(__APPLE__)
#define ENABLE_LOCKING_STYLE 1
#else
#define ENABLE_LOCKING_STYLE 0
#endif
#endif
#if ENABLE_LOCKING_STYLE && defined(__APPLE__)
	Jim_SetVar2(interp, "sqlite_options", "lock_proxy_pragmas", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "lock_proxy_pragmas", "0", JIM_GLOBAL);
#endif
#if defined(PREFER_PROXY_LOCKING) && defined(__APPLE__)
	Jim_SetVar2(interp, "sqlite_options", "prefer_proxy_locking", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "prefer_proxy_locking", "0", JIM_GLOBAL);
#endif

#ifdef OMIT_SHARED_CACHE
	Jim_SetVar2(interp, "sqlite_options", "shared_cache", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "shared_cache", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_SUBQUERY
	Jim_SetVar2(interp, "sqlite_options", "subquery", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "subquery", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_JIM_VARIABLE
	Jim_SetVar2(interp, "sqlite_options", "tclvar", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "tclvar", "1", JIM_GLOBAL);
#endif

	Jim_SetVar2(interp, "sqlite_options", "threadsafe", STRINGVALUE(THREADSAFE), JIM_GLOBAL);
	//_assert(sqlite3_threadsafe() == _THREADSAFE);

#ifdef OMIT_TEMPDB
	Jim_SetVar2(interp, "sqlite_options", "tempdb", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "tempdb", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_TRACE
	Jim_SetVar2(interp, "sqlite_options", "trace", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "trace", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_TRIGGER
	Jim_SetVar2(interp, "sqlite_options", "trigger", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "trigger", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_TRUNCATE_OPTIMIZATION
	Jim_SetVar2(interp, "sqlite_options", "truncate_opt", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "truncate_opt", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_UTF16
	Jim_SetVar2(interp, "sqlite_options", "utf16", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "utf16", "1", JIM_GLOBAL);
#endif

#if defined(OMIT_VACUUM) || defined(OMIT_ATTACH)
	Jim_SetVar2(interp, "sqlite_options", "vacuum", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "vacuum", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_VIEW
	Jim_SetVar2(interp, "sqlite_options", "view", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "view", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_VIRTUALTABLE
	Jim_SetVar2(interp, "sqlite_options", "vtab", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "vtab", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_WAL
	Jim_SetVar2(interp, "sqlite_options", "wal", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "wal", "1", JIM_GLOBAL);
#endif

#ifdef OMIT_WSD
	Jim_SetVar2(interp, "sqlite_options", "wsd", "0", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "wsd", "1", JIM_GLOBAL);
#endif

#if defined(SQLITE_ENABLE_UPDATE_DELETE_LIMIT) && !defined(SQLITE_OMIT_SUBQUERY)
	Jim_SetVar2(interp, "sqlite_options", "update_delete_limit", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "update_delete_limit", "0", JIM_GLOBAL);
#endif

#if defined(ENABLE_UNLOCK_NOTIFY)
	Jim_SetVar2(interp, "sqlite_options", "unlock_notify", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "unlock_notify", "0", JIM_GLOBAL);
#endif

#ifdef SECURE_DELETE
	Jim_SetVar2(interp, "sqlite_options", "secure_delete", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "secure_delete", "0", JIM_GLOBAL);
#endif

#ifdef MULTIPLEX_EXT_OVWR
	Jim_SetVar2(interp, "sqlite_options", "multiplex_ext_overwrite", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "multiplex_ext_overwrite", "0", JIM_GLOBAL);
#endif

#ifdef YYTRACKMAXSTACKDEPTH
	Jim_SetVar2(interp, "sqlite_options", "yytrackmaxstackdepth", "1", JIM_GLOBAL);
#else
	Jim_SetVar2(interp, "sqlite_options", "yytrackmaxstackdepth", "0", JIM_GLOBAL);
#endif

#define LINKVAR(x) { static const int cv_ ## x = x; Jim_LinkVar(interp, "SQLITE_" #x, (char *)&(cv_ ## x), JIM_LINK_INT|JIM_LINK_READ_ONLY); }
	LINKVAR(CORE_MAX_LENGTH);
	LINKVAR(MAX_COLUMN);
	LINKVAR(MAX_SQL_LENGTH);
	LINKVAR(MAX_EXPR_DEPTH);
	LINKVAR(MAX_COMPOUND_SELECT);
	LINKVAR(MAX_VDBE_OP);
	LINKVAR(MAX_FUNCTION_ARG);
	LINKVAR(MAX_VARIABLE_NUMBER);
	LINKVAR(MAX_PAGE_SIZE);
	LINKVAR(MAX_PAGE_COUNT);
	LINKVAR(MAX_LIKE_PATTERN_LENGTH);
	LINKVAR(MAX_TRIGGER_DEPTH);
	LINKVAR(DEFAULT_TEMP_CACHE_SIZE);
	LINKVAR(DEFAULT_CACHE_SIZE);
	LINKVAR(DEFAULT_PAGE_SIZE);
	LINKVAR(DEFAULT_FILE_FORMAT);
	LINKVAR(MAX_ATTACHED);
	LINKVAR(MAX_DEFAULT_PAGE_SIZE);
	{
		static const int cv_TEMP_STORE = TEMP_STORE;
		Jim_LinkVar(interp, "TEMP_STORE", (char *)&(cv_TEMP_STORE), JIM_LINK_INT|JIM_LINK_READ_ONLY);
	}

#ifdef _MSC_VER
	{
		static const int cv__MSC_VER = 1;
		Jim_LinkVar(interp, "_MSC_VER", (char *)&(cv__MSC_VER), JIM_LINK_INT|JIM_LINK_READ_ONLY);
	}
#endif
#ifdef __GNUC__
	{
		static const int cv___GNUC__ = 1;
		Jim_LinkVar(interp, "__GNUC__", (char *)&(cv___GNUC__), JIM_LINK_INT|JIM_LINK_READ_ONLY);
	}
#endif
}

// Register commands with the TCL interpreter.
__device__ int Sqliteconfig_Init(Jim_Interp *interp)
{
	set_options(interp);
	return JIM_OK;
}
