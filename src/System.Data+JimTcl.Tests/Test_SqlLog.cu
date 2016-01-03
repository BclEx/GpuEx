// OVERVIEW
//   This file contains experimental code used to record data from live SQLite applications that may be useful for offline analysis. Specifically:
//
//     1) The initial contents of all database files opened by the application, and
//
//     2) All SQL statements executed by the application.
//
// USAGE
//   To use this module, SQLite must be compiled with the SQLITE_ENABLE_SQLLOG pre-processor symbol defined and this file linked into the application somehow.
//
//   At runtime, logging is enabled by setting environment variable SQLITE_SQLLOG_DIR to the name of a directory in which to store logged 
//   data. The directory must already exist.
//
//   Usually, if the application opens the same database file more than once (either by attaching it or by using more than one database handle), only
//   a single copy is made. This behavior may be overridden (so that a separate copy is taken each time the database file is opened or attached)
//   by setting the environment variable SQLITE_SQLLOG_REUSE_FILES to 0.
//
// OUTPUT:
//   The SQLITE_SQLLOG_DIR is populated with three types of files:
//
//      sqllog_N.db   - Copies of database files. N may be any integer.
//
//      sqllog_N.sql  - A list of SQL statements executed by a single connection. N may be any integer.
//
//      sqllog.idx    - An index mapping from integer N to a database file name - indicating the full path of the
//                      database from which sqllog_N.db was copied.
//
// ERROR HANDLING:
//   This module attempts to make a best effort to continue logging if an IO or other error is encountered. For example, if a log file cannot 
//   be opened logs are not collected for that connection, but other logging proceeds as expected. Errors are logged by calling sqlite3_log().
#include <Core+Vdbe\Core+Vdbe.cu.h>
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "assert.h"
#include "sys/types.h"
#if OS_WIN
#include <process.h>
#endif
__device__ static int GetProcessId()
{
#if OS_WIN
	return (int)_getpid();
#else
	return (int)getpid();
#endif
}

#define ENVIRONMENT_VARIABLE1_NAME "SQLITE_SQLLOG_DIR"
#define ENVIRONMENT_VARIABLE2_NAME "SQLITE_SQLLOG_REUSE_FILES"

// Assume that all database and database file names are shorted than this.
#define SQLLOG_NAMESZ 512

// Maximum number of simultaneous database connections the process may open (if any more are opened an error is logged using sqlite3_log()
// and processing is halted).
#define MAX_CONNECTIONS 256

struct SLConn
{
	bool IsErr;                     // True if an error has occurred
	Context *Ctx;                   // Connection handle
	int Log;                      // First integer value used in file names
	FILE *Fd;						// File descriptor for log file
};

__device__ struct SLGlobal
{
	// Protected by MUTEX_STATIC_MASTER
	MutexEx Mutex;					// Recursive mutex

	// Protected by SLGlobal.mutex
	bool Reuse;                     // True to avoid extra copies of db files
	char Prefix[SQLLOG_NAMESZ];		// Prefix for all created files
	char Idx[SQLLOG_NAMESZ];		// Full path to *.idx file
	int NextLog;					// Used to allocate file names
	int NextDb;						// Used to allocate database file names
	bool Rec;                       // True if testSqllog() is called rec.
	int Clock;						// Clock value
	array_t3<int, SLConn, MAX_CONNECTIONS> Conns;
} _sqllogglobal;

// Return true if c is an ASCII whitespace character.
__device__ static int Sqllog_isspace(char c)
{
	return (c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r');
}

// The first argument points to a nul-terminated string containing an SQL command. Before returning, this function sets *pz to point to the start
// of the first token in this command, and *pn to the number of bytes in the token. This is used to check if the SQL command is an "ATTACH" or not.
__device__ static void SqllogTokenize(const char *z, const char **zOut, int *nOut)
{
	// Skip past any whitespace
	const char *p = z;
	while (Sqllog_isspace(*p)) { p++; }
	// Figure out how long the first token is
	*zOut = p;
	int n = 0;
	while ((p[n] >= 'a' && p[n] <= 'z') || (p[n] >= 'A' && p[n] <= 'Z')) n++;
	*nOut = n;
}

// Check if the logs directory already contains a copy of database file zFile. If so, return a pointer to the full path of the copy. Otherwise,
// return NULL.
//
// If a non-NULL value is returned, then the caller must arrange to eventually free it using sqlite3_free().
__device__ static char *SqllogFindFile(const char *file)
{
	// Open the index file for reading
	FILE *fd = _fopen(_sqllogglobal.Idx, "r");
	if (!fd)
	{
		SysEx_LOG(RC_IOERROR, "sqllogFindFile(): error in fopen()");
		return 0;
	}

	// Loop through each entry in the index file. If zFile is not NULL and the entry is a match, then set zRet to point to the filename of the existing
	// copy and break out of the loop.
	char *r = nullptr;
	while (_feof(fd) == 0)
	{
		char line[SQLLOG_NAMESZ*2+5];
		if (_fgets(line, sizeof(line), fd))
		{
			line[sizeof(line)-1] = '\0';
			char *z = line;
			while (*z >= '0' && *z <= '9') z++;
			while (*z == ' ') z++;

			int n = _strlen(z);
			while (n > 0 && Sqllog_isspace(z[n-1])) n--;

			if (n == _strlen(file) && !_memcmp(file, z, n))
			{
				char buf[16];
				_memset(buf, 0, sizeof(buf));
				z = line;
				while (*z >= '0' && *z <= '9') { buf[z-line] = *z; z++; }
				r = _mprintf("%s_%s.db", _sqllogglobal.Prefix, buf);
				break;
			}
		}
	}

	if (_ferror(fd))
		SysEx_LOG(RC_IOERROR, "sqllogFindFile(): error reading index file");

	_fclose(fd);
	return r;
}

__device__ static RC SqllogFindAttached(SLConn *p, const char *search, char *name, char *file)
{
	// The "PRAGMA database_list" command returns a list of databases in the order that they were attached. So a newly attached database is 
	// described by the last row returned.
	_assert(!_sqllogglobal.Rec);
	_sqllogglobal.Rec = true;
	Vdbe *stmt;
	RC rc = Prepare::Prepare_v2(p->Ctx, "PRAGMA database_list", -1, &stmt, nullptr);
	if (rc == RC_OK)
	{
		while (stmt->Step() == RC_ROW)
		{
			const char *val1 = (const char *)Vdbe::Column_Text(stmt, 1);
			int val1Length = Vdbe::Column_Bytes(stmt, 1);
			_memcpy(name, val1, val1Length+1);

			const char *val2 = (const char *)Vdbe::Column_Text(stmt, 2);
			int val2Length = Vdbe::Column_Bytes(stmt, 2);
			_memcpy(file, val2, val2Length+1);

			if (search && _strlen(search) == val1Length && !_strncmp(search, val1, val1Length))
				break;
		}
		rc = Vdbe::Finalize(stmt);
	}
	_sqllogglobal.Rec = false;

	if (rc != RC_OK)
		SysEx_LOG(rc, "sqllogFindAttached(): error in \"PRAGMA database_list\"");
	return rc;
}

// Parameter zSearch is the name of a database attached to the database connection associated with the first argument. This function creates
// a backup of this database in the logs directory.
//
// The name used for the backup file is automatically generated. Call it zFile.
//
// If the bLog parameter is true, then a statement of the following form is written to the log file associated with *p:
//    ATTACH 'zFile' AS 'zName';
//
// Otherwise, if bLog is false, a comment is added to the log file:
//    -- Main database file is 'zFile'
//
// The SLGlobal.mutex mutex is always held when this function is called.
__device__ static void SqllogCopydb(SLConn *p, const char *search, bool log)
{
	char name[SQLLOG_NAMESZ];      // Attached database name
	char file[SQLLOG_NAMESZ];      // Database file name
	RC rc = SqllogFindAttached(p, search, name, file);
	if (rc != RC_OK) return;

	char *init = nullptr;
	if (file[0] == '\0')
		init = _mprintf("");
	else
	{
		init = (_sqllogglobal.Reuse ? SqllogFindFile(file) : 0);
		if (!init)
		{
			// Generate a file-name to use for the copy of this database
			int db = _sqllogglobal.NextDb++;
			init = _mprintf("%s_%d.db", _sqllogglobal.Prefix, db);

			// Create the backup
			_assert(!_sqllogglobal.Rec);
			_sqllogglobal.Rec = true;
			Context *copy = nullptr;
			rc = Main::Open(init, &copy);
			if (rc == RC_OK)
			{
				Main::Exec(copy, "PRAGMA synchronous = 0", 0, 0, 0);
				Backup *bak = Backup::Init(copy, "main", p->Ctx, name);
				if (bak)
				{
					bak->Step(-1);
					rc = Backup::Finish(bak);
				}
				else
					rc = Main::ErrCode(copy);
				Main::Close(copy);
			}
			_sqllogglobal.Rec = false;

			if (rc == RC_OK)
			{
				// Write an entry into the database index file
				FILE *fd = _fopen(_sqllogglobal.Idx, "a");
				if (fd)
				{
					_fprintf(fd, "%d %s\n", db, file);
					_fclose(fd);
				}
			}
			else
				SysEx_LOG(rc, "sqllogCopydb(): error backing up database");
		}
	}

	char *free;
	if (log)
		free = _mprintf("ATTACH '%q' AS '%q'; -- clock=%d\n", init, name, _sqllogglobal.Clock++);
	else
		free = _mprintf("-- Main database is '%q'\n", init);
	_fprintf(p->Fd, "%s", free);
	_free(free);

	_free(init);
}

// If it is not already open, open the log file for connection *p. 
//
// The SLGlobal.mutex mutex is always held when this function is called.
__device__ static void SqllogOpenlog(SLConn *p)
{
	// If the log file has not yet been opened, open it now.
	if (!p->Fd)
	{
		// If it is still NULL, have global.zPrefix point to a copy of environment variable $ENVIRONMENT_VARIABLE1_NAME.
		if (_sqllogglobal.Prefix[0] == 0)
		{
			char *var = getenv(ENVIRONMENT_VARIABLE1_NAME);
			if (!var || _strlen(var)+10 >= (sizeof(_sqllogglobal.Prefix))) return;
			_sprintf(_sqllogglobal.Prefix, "%s/sqllog_%d", var, GetProcessId());
			_sprintf(_sqllogglobal.Idx, "%s.idx", _sqllogglobal.Prefix);
			if (getenv(ENVIRONMENT_VARIABLE2_NAME))
				_sqllogglobal.Reuse = (_atoi(getenv(ENVIRONMENT_VARIABLE2_NAME)) != 0);
			FILE *fd = _fopen(_sqllogglobal.Idx, "w");
			if (fd) _fclose(fd);
		}

		// Open the log file
		char *log = _mprintf("%s_%d.sql", _sqllogglobal.Prefix, p->Log);
		p->Fd = _fopen(log, "w");
		_free(log);
		if (!p->Fd)
			SysEx_LOG(RC_IOERR, "sqllogOpenlog(): Failed to open log file");
	}
}

// This function is called if the SQLLOG callback is invoked to report execution of an SQL statement. Parameter p is the connection the statement
// was executed by and parameter zSql is the text of the statement itself.
__device__ static void TestSqllogStmt(SLConn *p, const char *sql)
{
	const char *first;		// Pointer to first token in zSql
	int firstLength;		// Size of token zFirst in bytes
	SqllogTokenize(sql, &first, &firstLength);
	if (firstLength != 6 || _strncmp("ATTACH", first, 6)) // Not an ATTACH statement. Write this directly to the log.
		_fprintf(p->Fd, "%s; -- clock=%d\n", sql, _sqllogglobal.Clock++);
	else
		SqllogCopydb(p, 0, 1); // This is an ATTACH statement. Copy the database.
}

// The SQLITE_CONFIG_SQLLOG callback registered by sqlite3_init_sqllog().
__device__ static void TestSqllog(void *notUsed, Context *ctx, const char *sql, int type)
{
	SLConn *p = nullptr;
	MutexEx master = _mutex_alloc(MUTEX_STATIC_MASTER);

	_assert(type == 0 || type == 1 || type == 2);
	_assert((type == 2) == (sql == 0));

	// This is a database open command.
	if (type == 0)
	{
		_mutex_enter(master);
		if (!_sqllogglobal.Mutex)
			_sqllogglobal.Mutex = _mutex_alloc(MUTEX_RECURSIVE);
		p = &_sqllogglobal.Conns[_sqllogglobal.Conns.length++];
		p->Fd = 0;
		p->Ctx = ctx;
		p->Log = _sqllogglobal.NextLog++;
		_mutex_leave(master);

		// Open the log and take a copy of the main database file
		_mutex_enter(_sqllogglobal.Mutex);
		if (!_sqllogglobal.Rec)
		{
			SqllogOpenlog(p);
			if (p->Fd) SqllogCopydb(p, "main", false);
		}
		_mutex_leave(_sqllogglobal.Mutex);
	}
	else
	{
		int i;
		for (i = 0; i < _sqllogglobal.Conns.length; i++)
		{
			p = &_sqllogglobal.Conns[i];
			if (p->Ctx == ctx) break;
		}
		if (i == _sqllogglobal.Conns.length) return;

		// A database handle close command
		if (type == 2)
		{
			_mutex_enter(master);
			if (p->Fd) _fclose(p->Fd);
			p->Ctx = nullptr;
			p->Fd = nullptr;

			_sqllogglobal.Conns.length--;
			if (_sqllogglobal.Conns.length == 0)
			{
				_mutex_free(_sqllogglobal.Mutex);
				_sqllogglobal.Mutex = nullptr;
			}
			else
			{
				int shift = (int)(&_sqllogglobal.Conns[_sqllogglobal.Conns.length] - p);
				if (shift > 0)
					_memmove(p, &p[1], shift*sizeof(SLConn));
			}
			_mutex_leave(master);

			// An ordinary SQL command.
		}
		else if (p->Fd)
		{
			_mutex_enter(_sqllogglobal.Mutex);
			if (!_sqllogglobal.Rec)
				TestSqllogStmt(p, sql);
			_mutex_leave(_sqllogglobal.Mutex);
		}
	}
}

// This function is called either before sqlite3_initialized() or by it. It checks if the SQLITE_SQLLOG_DIR variable is defined, and if so 
// registers an SQLITE_CONFIG_SQLLOG callback to record the applications database activity.
__device__ void sqlite3_init_sqllog()
{
	if (getenv(ENVIRONMENT_VARIABLE1_NAME))
	{
		if (SysEx::Config(SysEx::CONFIG_SQLLOG, TestSqllog, 0) == RC_OK)
		{
			_memset(&_sqllogglobal, 0, sizeof(_sqllogglobal));
			_sqllogglobal.Reuse = true;
		}
	}
}
