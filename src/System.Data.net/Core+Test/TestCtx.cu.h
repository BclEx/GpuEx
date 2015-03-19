#include "..\Core+Vdbe\Core+Vdbe.cu.h"
#include "TestTcl.cu.h"

typedef struct TestCtx TestCtx;

// New SQL functions can be created as TCL scripts.  Each such function is described by an instance of the following structure.
typedef struct TestSqlFunc TestSqlFunc;
struct SqlFunc
{
	Tcl_Interp *Interp;		// The TCL interpret to execute the function
	Tcl_Obj *Script;		// The Tcl_Obj representation of the script
	TestCtx *Ctx;		// Database connection that owns this function
	bool UseEvalObjv;	// True if it is safe to use Tcl_EvalObjv
	char *Name;			// Name of this function
	SqlFunc *Next;		// Next function on the list of them all
};

// New collation sequences function can be created as TCL scripts.  Each such function is described by an instance of the following structure.
typedef struct SqlCollate SqlCollate;
struct SqlCollate
{
	Tcl_Interp *Interp;		// The TCL interpret to execute the function
	char *Script;       // The script to be run
	SqlCollate *Next;   // Next function on the list of them all
};

// Prepared statements are cached for faster execution.  Each prepared statement is described by an instance of the following structure.
typedef struct SqlPreparedStmt SqlPreparedStmt;
struct SqlPreparedStmt
{
	SqlPreparedStmt *Next;  // Next in linked list
	SqlPreparedStmt *Prev;  // Previous on the list
	Vdbe *Stmt;				// The prepared statement
	int SqlLength;          // chars in zSql[]
	const char *Sql;		// Text of the SQL statement
	array_t<Tcl_Obj *> Parms;	// Array of referenced object pointers
};

typedef struct IncrblobChannel IncrblobChannel;

struct TestCtx
{
	Context *Ctx;				// The "real" database structure. MUST BE FIRST
	Tcl_Interp *Interp;			// The interpreter used for this database
	char *Busy;					// The busy callback routine
	char *Commit;				// The commit hook callback routine
	char *Trace;				// The trace callback routine
	char *Profile;				// The profile callback routine
	char *Progress;				// The progress callback routine
	char *Auth;					// The authorization callback routine
	int DisableAuth;			// Disable the authorizer if it exists
	char *NullText;				// Text to substitute for an SQL NULL value
	SqlFunc *Funcs;				// List of SQL functions
	Tcl_Obj *UpdateHook;		// Update hook script (if any)
	Tcl_Obj *RollbackHook;		// Rollback hook script (if any)
	Tcl_Obj *WalHook;			// WAL hook script (if any)
	Tcl_Obj *UnlockNotify;		// Unlock notify script (if any)
	SqlCollate *Collates;		// List of SQL collation functions
	RC RC;						// Return code of most recent sqlite3_exec()
	Tcl_Obj *CollateNeeded;		// Collation needed script
	array_t<SqlPreparedStmt> Stmts;	// List of prepared statements
	SqlPreparedStmt *StmtLast;		// Last statement in the list
	int MaxStmt;				// The next maximum number of stmtList
	IncrblobChannel *Incrblobs;	// Linked list of open incrblob channels
	int Steps, Sorts, Indexs;	// Statistics for most recent operation
	int Transactions;			// Number of nested [transaction] methods
#ifdef TEST
	bool LegacyPrepare;			// True to use sqlite3_prepare()
#endif

	__device__ ::RC DB_AUTHORIZER(int objc, Tcl_Obj *const*objv);
	__device__ ::RC DB_BACKUP(int objc, Tcl_Obj *const*objv);
	__device__ ::RC DB_BUSY(int objc, Tcl_Obj *const*objv);
	__device__ ::RC DB_CACHE(int objc, Tcl_Obj *const*objv);
	__device__ ::RC DB_CHANGES(int objc, Tcl_Obj *const*objv);
	__device__ void DB_CLOSE(int objc, Tcl_Obj *const*objv);

	__device__ ::RC DB_EVAL(int objc, Tcl_Obj *const*objv);

};

struct IncrblobChannel
{
	Blob *Blob;					// sqlite3 blob handle
	TestCtx *TCtx;				// Associated database connection
	int SeekId;					// Current seek offset
	Tcl_Channel Channel;		// Channel identifier
	IncrblobChannel *Next;		// Linked list of all open incrblob channels
	IncrblobChannel *Prev;		// Linked list of all open incrblob channels
};
