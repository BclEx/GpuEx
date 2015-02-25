namespace Core
{
	#pragma region Authorization

	enum ARC : uint8
	{
		ARC_OK = 0,			// Successful result
		ARC_DENY = 1,		// Abort the SQL statement with an error
		ARC_IGNORE = 2,		// Don't allow access, but don't generate an error
	};

	enum AUTH : uint8
	{
		AUTH_CREATE_INDEX        =  1,   // Index Name      Table Name
		AUTH_CREATE_TABLE        =  2,   // Table Name      NULL
		AUTH_CREATE_TEMP_INDEX   =  3,   // Index Name      Table Name
		AUTH_CREATE_TEMP_TABLE   =  4,   // Table Name      NULL
		AUTH_CREATE_TEMP_TRIGGER =  5,   // Trigger Name    Table Name
		AUTH_CREATE_TEMP_VIEW    =  6,   // View Name       NULL
		AUTH_CREATE_TRIGGER      =  7,   // Trigger Name    Table Name
		AUTH_CREATE_VIEW         =  8,   // View Name       NULL
		AUTH_DELETE              =  9,   // Table Name      NULL
		AUTH_DROP_INDEX          = 10,   // Index Name      Table Name
		AUTH_DROP_TABLE          = 11,   // Table Name      NULL
		AUTH_DROP_TEMP_INDEX     = 12,   // Index Name      Table Name
		AUTH_DROP_TEMP_TABLE     = 13,   // Table Name      NULL
		AUTH_DROP_TEMP_TRIGGER   = 14,   // Trigger Name    Table Name
		AUTH_DROP_TEMP_VIEW      = 15,   // View Name       NULL
		AUTH_DROP_TRIGGER        = 16,   // Trigger Name    Table Name
		AUTH_DROP_VIEW           = 17,   // View Name       NULL
		AUTH_INSERT              = 18,   // Table Name      NULL
		AUTH_PRAGMA              = 19,   // Pragma Name     1st arg or NULL
		AUTH_READ                = 20,   // Table Name      Column Name
		AUTH_SELECT              = 21,   // NULL            NULL
		AUTH_TRANSACTION         = 22,   // Operation       NULL
		AUTH_UPDATE              = 23,   // Table Name      Column Name
		AUTH_ATTACH              = 24,   // Filename        NULL
		AUTH_DETACH              = 25,   // Database Name   NULL
		AUTH_ALTER_TABLE         = 26,   // Database Name   Table Name
		AUTH_REINDEX             = 27,   // Index Name      NULL
		AUTH_ANALYZE             = 28,   // Table Name      NULL
		AUTH_CREATE_VTABLE       = 29,   // Table Name      Module Name
		AUTH_DROP_VTABLE         = 30,   // Table Name      Module Name
		AUTH_FUNCTION            = 31,   // NULL            Function Name
		AUTH_SAVEPOINT           = 32,   // Operation       Savepoint Name 
		AUTH_COPY                =  0,   // No longer used
	};

	struct Auth
	{
		__device__ static RC SetAuthorizer(Context *ctx, ARC (*auth)(void*,int,const char*,const char*,const char*,const char*), void *args);
		__device__ static void BadReturnCode(Parse *parse);
		__device__ static ARC ReadColumn(Parse *parse, const char *table, const char *column, int db);
		__device__ static void Read(Parse *parse, Expr *expr, Schema *schema, SrcList *tableList);
		__device__ static ARC Check(Parse *parse, AUTH code, const char *arg1, const char *arg2, const char *arg3);
		__device__ static void ContextPush(Parse *parse, AuthContext *actx, const char *context);
		__device__ static void Auth::ContextPop(AuthContext *actx);
	};

#pragma endregion

	enum LIMIT : uint8
	{
		LIMIT_LENGTH = 0,
		LIMIT_SQL_LENGTH = 1,
		LIMIT_COLUMN = 2,
		LIMIT_EXPR_DEPTH = 3,
		LIMIT_COMPOUND_SELECT = 4,
		LIMIT_VDBE_OP = 5,
		LIMIT_FUNCTION_ARG = 6,
		LIMIT_ATTACHED = 7,
		LIMIT_LIKE_PATTERN_LENGTH = 8,
		LIMIT_VARIABLE_NUMBER = 9,
		LIMIT_TRIGGER_DEPTH = 10,
		LIMIT_MAX_ = 11,
	};

	enum MAGIC : uint32
	{
		MAGIC_OPEN = 0xa029a697,	// Database is open
		MAGIC_CLOSED = 0x9f3c2d33,  // Database is closed
		MAGIC_SICK = 0x4b771290,	// Error and awaiting close
		MAGIC_BUSY = 0xf03b7906,	// Database currently in use
		MAGIC_ERROR = 0xb5357930,	// An SQLITE_MISUSE error occurred
		MAGIC_ZOMBIE = 0x64cffc7f,  // Close with last statement close
	};

#ifndef OMIT_BUILTIN_TEST
#define CtxOptimizationDisabled(ctx, mask)  (((ctx)->OptFlags&(mask))!=0)
#define CtxOptimizationEnabled(ctx, mask)   (((ctx)->OptFlags&(mask))==0)
#else
#define CtxOptimizationDisabled(ctx, mask)  0
#define CtxOptimizationEnabled(ctx, mask)   1
#endif

	enum OPTFLAG : uint16
	{
		OPTFLAG_QueryFlattener = 0x0001,	// Query flattening
		OPTFLAG_ColumnCache = 0x0002,		// Column cache
		OPTFLAG_GroupByOrder = 0x0004,		// GROUPBY cover of ORDERBY
		OPTFLAG_FactorOutConst = 0x0008,	// Constant factoring
		OPTFLAG_IdxRealAsInt = 0x0010,		// Store REAL as INT in indices
		OPTFLAG_DistinctOpt = 0x0020,		// DISTINCT using indexes
		OPTFLAG_CoverIdxScan = 0x0040,		// Covering index scans
		OPTFLAG_OrderByIdxJoin = 0x0080,	// ORDER BY of joins via index
		OPTFLAG_SubqCoroutine = 0x0100,		// Evaluate subqueries as coroutines
		OPTFLAG_Transitive = 0x0200,		// Transitive constraints
		OPTFLAG_AllOpts = 0xffff,			// All optimizations
	};

	struct FuncDefHash
	{
		FuncDef *data[23]; // Hash table for functions
	};

	struct VTable;
	struct VTableContext;
	class Context : public BContext
	{
	public:
		struct InitInfo
		{
			int NewTid;						// Rootpage of table being initialized
			uint8 DB;						// Which db file is being initialized
			bool Busy;						// TRUE if currently initializing
			bool OrphanTrigger;				// Last statement is orphaned TEMP trigger
		};

		VSystem *Vfs;						// OS Interface
		Vdbe *Vdbes;						// List of active virtual machines
		CollSeq *DefaultColl;				// The default collating sequence (BINARY)
		//TAGBASE::Mutex
		//BCONTEXT::DBs
		//BCONTEXT::Flags
		int64 LastRowID;					// ROWID of most recent insert (see above)
		VSystem::OPEN OpenFlags;			// Flags passed to sqlite3_vfs.xOpen()
		//TAGBASE::ErrCode
		//TAGBASE::ErrMask
		OPTFLAG OptFlags;					// Flags to enable/disable optimizations
		uint8 AutoCommit;					// The auto-commit flag.
		uint8 TempStore;					// 1: file 2: memory 0: default
		//TAGBASE::MallocFailed
		IPager::LOCKINGMODE DefaultLockMode;// Default locking-mode for attached dbs
		Btree::AUTOVACUUM NextAutovac;		// Autovac setting after VACUUM if >=0
		uint8 SuppressErr;					// Do not issue error messages if true
		uint8 VTableOnConflict;				// Value to return for s3_vtab_on_conflict()
		uint8 IsTransactionSavepoint;		// True if the outermost savepoint is a TS
		int NextPagesize;					// Pagesize after VACUUM if >0
		MAGIC Magic;						// Magic number for detect library misuse
		int Changes;						// Value returned by sqlite3_changes()
		int TotalChanges;					// Value returned by sqlite3_total_changes()
		int Limits[LIMIT_MAX_];				// Limits
		InitInfo Init;						// Information used during initialization
		//BCONTEXT::ActiveVdbeCnt
		int WriteVdbeCnt;					// Number of active VDBEs that are writing
		int VdbeExecCnt;					// Number of nested calls to VdbeExec()
		array_t<void *>Extensions;          // Array of shared library handles
		void (*Trace)(void*,const char*);   // Trace function
		void *TraceArg;                     // Argument to the trace function
		void (*Profile)(void*,const char*,uint64);  // Profiling function
		void *ProfileArg;                   // Argument to profile function
		void *CommitArg;					// Argument to xCommitCallback()
		RC (*CommitCallback)(void*);		// Invoked at every commit.
		void *RollbackArg;					// Argument to xRollbackCallback()
		void (*RollbackCallback)(void*);	// Invoked at every commit.
		void *UpdateArg;
		void (*UpdateCallback)(void*,int, const char*,const char*,int64);
#ifndef OMIT_WAL
		int (*WalCallback)(void*,Context*,const char*,int);
		void *WalArg;
#endif
		void(*CollNeeded)(void*,Context*,TEXTENCODE,const char*);
		void(*CollNeeded16)(void*,Context*,TEXTENCODE,const void*);
		void *CollNeededArg;
		Mem *Err;							// Most recent error message
		char *ErrMsg;						// Most recent error message (UTF-8 encoded)
		char *ErrMsg16;						// Most recent error message (UTF-16 encoded)
		union
		{
			volatile bool IsInterrupted;	// True if sqlite3_interrupt has been called
			double NotUsed1;				// Spacer
		} u1;
		//TAGBASE::Lookaside Lookaside;		// Lookaside malloc configuration
#ifndef OMIT_AUTHORIZATION
		ARC (*Auth)(void*,int,const char*,const char*,const char*,const char*); // Access authorization function
		void *AuthArg;						// 1st argument to the access auth function
#endif
#ifndef OMIT_PROGRESS_CALLBACK
		int (*Progress)(void*);			// The progress callback
		void *ProgressArg;					// Argument to the progress callback
		int ProgressOps;					// Number of opcodes for progress callback
#endif
#ifndef OMIT_VIRTUALTABLE
		Hash Modules;						// populated by sqlite3_create_module()
		VTableContext *VTableCtx;			// Context for active vtab connect/create
		array_t<VTable *> VTrans;			// Virtual tables with open transactions / Allocated size of aVTrans
		VTable *Disconnect;					// Disconnect these in next sqlite3_prepare()
#endif
		FuncDefHash Funcs;					// Hash table of connection functions
		Hash CollSeqs;						// All collating sequences
		//BCONTEXT::BusyHandler
		//BCONTEXT::DbStatics
		//BCONTEXT::Savepoints
		//BCONTEXT::BusyTimeout
		//BCONTEXT::SavepointsLength
		int Statements;						// Number of nested statement-transactions
		int64 DeferredCons;					// Net deferred constraints this transaction.
		int *BytesFreed;					// If not NULL, increment this in DbFree()
#ifdef ENABLE_UNLOCK_NOTIFY
		//BCONTEXT::BlockingConnection
		//BCONTEXT::UnlockConnection
		//BCONTEXT::UnlockArg
		//BCONTEXT::UnlockNotify
		//BCONTEXT::NextBlocked
#endif
	};

}