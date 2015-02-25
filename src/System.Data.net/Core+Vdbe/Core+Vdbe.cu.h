#include "../Core+Btree/Core+Btree.cu.h"
#include "Parse+Parser.orig.h"
#include "../Opcodes.h"
#include <stdio.h>
namespace Core
{
#pragma region Limit Types
#if MAX_ATTACHED > 30
	typedef uint64 yDbMask;
#else
	typedef unsigned int yDbMask;
#endif
#if MAX_VARIABLE_NUMBER <= 32767
	typedef int16 yVars;
#else
	typedef int yVars;
#endif
#if _64BITSTATS
	typedef uint64 tRowcnt;    // 64-bit only if requested at compile-time
#else
	typedef uint32 tRowcnt;    // 32-bit is the default
#endif
	typedef uint64 Bitmask;
#pragma endregion 

#pragma region Limits

#ifndef N_COLCACHE
#define N_COLCACHE 10
#endif
#define BMS ((int)(sizeof(Bitmask)*8))

	// from sqliteInt.h
#ifdef OMIT_TEMPDB
#define E_OMIT_TEMPDB true
#else
#define E_OMIT_TEMPDB false
#endif
#define MAX_FILE_FORMAT 4
#ifndef DEFAULT_FILE_FORMAT
#define DEFAULT_FILE_FORMAT 4
#endif
#ifndef DEFAULT_RECURSIVE_TRIGGERS
#define DEFAULT_RECURSIVE_TRIGGERS 0
#endif
#ifndef TEMP_STORE
#define TEMP_STORE 1
#endif

	// from sqliteLimits.h
#ifndef MAX_COLUMN
#define MAX_COLUMN 2000
#endif
#ifndef MAX_SQL_LENGTH
#define MAX_SQL_LENGTH 1000000000
#endif
#ifndef MAX_EXPR_DEPTH
#define MAX_EXPR_DEPTH 1000
#endif
#ifndef MAX_COMPOUND_SELECT
#define MAX_COMPOUND_SELECT 500
#endif
#ifndef MAX_VDBE_OP
#define MAX_VDBE_OP 25000
#endif
#ifndef MAX_FUNCTION_ARG
#define MAX_FUNCTION_ARG 127
#endif

	//#ifndef DEFAULT_CACHE_SIZE
	//#define DEFAULT_CACHE_SIZE  2000
	//#endif
#ifndef DEFAULT_TEMP_CACHE_SIZE
#define DEFAULT_TEMP_CACHE_SIZE  500
#endif
#ifndef DEFAULT_WAL_AUTOCHECKPOINT
#define DEFAULT_WAL_AUTOCHECKPOINT  1000
#endif

#ifndef MAX_ATTACHED
#define MAX_ATTACHED 10
#endif
#ifndef MAX_VARIABLE_NUMBER
#define MAX_VARIABLE_NUMBER 999
#endif
	//#ifdef MAX_PAGE_SIZE
	//#undef MAX_PAGE_SIZE
	//#endif
	//#define MAX_PAGE_SIZE 65536

#ifndef DEFAULT_PAGE_SIZE
#define DEFAULT_PAGE_SIZE 1024
#endif
#if DEFAULT_PAGE_SIZE > MAX_PAGE_SIZE
#undef DEFAULT_PAGE_SIZE
#define DEFAULT_PAGE_SIZE MAX_PAGE_SIZE
#endif

#ifndef MAX_DEFAULT_PAGE_SIZE
#define MAX_DEFAULT_PAGE_SIZE 8192
#endif
#if MAX_DEFAULT_PAGE_SIZE > MAX_PAGE_SIZE
#undef MAX_DEFAULT_PAGE_SIZE
#define MAX_DEFAULT_PAGE_SIZE MAX_PAGE_SIZE
#endif
#ifndef MAX_PAGE_COUNT
#define MAX_PAGE_COUNT 1073741823
#endif
#ifndef MAX_LIKE_PATTERN_LENGTH
#define MAX_LIKE_PATTERN_LENGTH 50000
#endif
#ifndef MAX_TRIGGER_DEPTH
#define MAX_TRIGGER_DEPTH 1000
#endif

#pragma endregion 

#pragma region Func

	enum FUNC : uint8
	{
		FUNC_LIKE = 0x01,			// Candidate for the LIKE optimization
		FUNC_CASE = 0x02,			// Case-sensitive LIKE-type function
		FUNC_EPHEM = 0x04,			// Ephemeral.  Delete with VDBE
		FUNC_NEEDCOLL = 0x08,		// sqlite3GetFuncCollSeq() might be called
		FUNC_COUNT = 0x10,			// Built-in count(*) aggregate
		FUNC_COALESCE = 0x20,		// Built-in coalesce() or ifnull() function
		FUNC_LENGTH = 0x40,			// Built-in length() function
		FUNC_TYPEOF = 0x80,			// Built-in typeof() function
	};
	__device__ inline void operator|=(FUNC &a, int b) { a = (FUNC)(a | b); }
	__device__ inline FUNC operator|(FUNC a, FUNC b) { return (FUNC)((int)a | (int)b); }

	struct FuncDestructor
	{
		int Refs;
		void (*Destroy)(void *);
		void *UserData;
	};

	struct FuncContext;
	struct FuncDef
	{
		int16 Args;					// Number of arguments.  -1 means unlimited
		TEXTENCODE PrefEncode;		// Preferred text encoding (SQLITE_UTF8, 16LE, 16BE)
		FUNC Flags;					// Some combination of SQLITE_FUNC_*
		void *UserData;				// User data parameter
		FuncDef *Next;				// Next function with same name */
		void (*Func)(FuncContext *, int, Mem**); // Regular function
		void (*Step)(FuncContext *, int, Mem**); // Aggregate step
		void (*Finalize)(FuncContext *); // Aggregate finalizer
		char *Name;					// SQL name of the function.
		FuncDef *Hash;				// Next with a different name but the same hash
		FuncDestructor *Destructor; // Reference counted destructor function
	};
	struct FuncDefHash;

	//   FUNCTION(name, args, arg, nc, func)
	//     Used to create a scalar function definition of a function zName implemented by C function func that accepts nArg arguments. The
	//     value passed as arg is cast to a (void*) and made available as the user-data (sqlite3_user_data()) for the function. If 
	//     argument nc is true, then the SQLITE_FUNC_NEEDCOLL flag is set.
	//
	//   AGGREGATE(name, args, arg, nc, step, final)
	//     Used to create an aggregate function definition implemented by the C functions step and final. The first four parameters
	//     are interpreted in the same way as the first 4 parameters to FUNCTION().
	//
	//   LIKEFUNC(name, args, arg, flags)
	//     Used to create a scalar function definition of a function zName that accepts nArg arguments and is implemented by a call to C 
	//     function likeFunc. Argument arg is cast to a (void *) and made available as the function user-data (sqlite3_user_data()). The
	//     FuncDef.flags variable is set to the value passed as the flags parameter.
#define FUNCTION(name, args, arg, nc, func) {args, TEXTENCODE_UTF8, (FUNC)(nc*FUNC_NEEDCOLL), INT_TO_PTR(arg), 0, func, 0, 0, #name, 0, 0}
#define FUNCTION2(name, args, arg, nc, func, extraFlags) {args, TEXTENCODE_UTF8, (FUNC)(nc*FUNC_NEEDCOLL)|extraFlags, INT_TO_PTR(arg), 0, func, 0, 0, #name, 0, 0}
#define STR_FUNCTION(name, args, arg, nc, func) {args, TEXTENCODE_UTF8, (FUNC)(nc*FUNC_NEEDCOLL), arg, 0, func, 0, 0, #name, 0, 0}
#define LIKEFUNC(name, args, arg, flags) {args, TEXTENCODE_UTF8, (FUNC)flags, (void *)arg, 0, LikeFunc, 0, 0, #name, 0, 0}
#define AGGREGATE(name, args, arg, nc, step, final) {args, TEXTENCODE_UTF8, (FUNC)(nc*FUNC_NEEDCOLL), INT_TO_PTR(arg), 0, 0, step,final,#name,0,0}

#pragma endregion

#pragma region ITable

	class Context;
	struct IIndexInfo;
	struct IVTable;
	struct IVTableCursor;
	struct ITableModule;

	struct TableModule
	{
		const ITableModule *IModule;    // Callback pointers
		const char *Name;               // Name passed to create_module()
		void *Aux;                      // pAux passed to create_module()
		void (*Destroy)(void *);        // Module destructor function
	};

	struct ITableModule //was:sqlite3_module
	{
	public:
		int Version;
		RC (*Create)(Context *, void *aux, int argc, const char *const *argv, IVTable **vtabs, char **);
		RC (*Connect)(Context *, void *aux, int argc, const char *const *argv, IVTable **vtabs, char **);
		RC (*BestIndex)(IVTable *vtab, IIndexInfo *);
		RC (*Disconnect)(IVTable *vtab);
		RC (*Destroy)(IVTable *vtab);
		RC (*Open)(IVTable *vtab, IVTableCursor **cursors);
		RC (*Close)(IVTableCursor*);
		RC (*Filter)(IVTableCursor*, int idxNum, const char *idxStr, int argc, Mem **argv);
		RC (*Next)(IVTableCursor*);
		RC (*Eof)(IVTableCursor*);
		RC (*Column)(IVTableCursor *, FuncContext *, int);
		RC (*Rowid)(IVTableCursor *, int64 *rowid);
		RC (*Update)(IVTable *, int, Mem **, int64 *);
		RC (*Begin)(IVTable *vtab);
		RC (*Sync)(IVTable *vtab);
		RC (*Commit)(IVTable *vtab);
		RC (*Rollback)(IVTable *vtab);
		RC (*FindFunction)(IVTable *vtab, int argsLength, const char *name, void (**func)(FuncContext *, int, Mem **), void **args);
		RC (*Rename)(IVTable *vtab, const char *new_);
		RC (*Savepoint)(IVTable *vtab, int);
		RC (*Release)(IVTable *vtab, int);
		RC (*RollbackTo)(IVTable *vtab, int);
	};

	enum INDEX_CONSTRAINT : uint8
	{
		INDEX_CONSTRAINT_EQ = 2,
		INDEX_CONSTRAINT_GT = 4,
		INDEX_CONSTRAINT_LE = 8,
		INDEX_CONSTRAINT_LT = 16,
		INDEX_CONSTRAINT_GE = 32,
		INDEX_CONSTRAINT_MATCH = 64,
	};

	struct IIndexInfo // was:sqlite3_index_info
	{
		struct Constraint
		{
			int Column;				// Column on left-hand side of constraint
			INDEX_CONSTRAINT OP;	// Constraint operator
			bool Usable;			// True if this constraint is usable
			int TermOffset;			// Used internally - xBestIndex should ignore
		};
		struct Orderby
		{
			int Column;				// Column number
			bool Desc;				// True for DESC.  False for ASC.
		};
		struct ConstraintUsage
		{
			int ArgvIndex;			// if >0, constraint is part of argv to xFilter
			unsigned char Omit;		// Do not code a test for this constraint
		};
		// INPUTS
		array_t<Constraint> Constraints; // Table of WHERE clause constraints
		array_t<Orderby> OrderBys;	// The ORDER BY clause
		// OUTPUTS
		array_t<ConstraintUsage> ConstraintUsages;
		int IdxNum;					// Number used to identify the index
		char *IdxStr;				// String, possibly obtained from sqlite3_malloc
		bool NeedToFreeIdxStr;		// Free idxStr using sqlite3_free() if true
		bool OrderByConsumed;		// True if output is already ordered
		double EstimatedCost;		// Estimated cost of using this index
	};

	struct IVTable //was:sqlite3_vtab
	{
		const ITableModule *IModule;// The module for this virtual table
		char *ErrMsg;				// Error message from sqlite3_mprintf()
		// Virtual table implementations will typically add additional fields
	};

	struct IVTableCursor //was:sqlite3_vtab_cursor
	{
		IVTable *IVTable;			// Virtual table of this cursor
		// Virtual table implementations will typically add additional fields
	};

#pragma endregion

#pragma region Types

	enum OE : uint8
	{
		OE_None = 0,   // There is no constraint to check
		OE_Rollback = 1,   // Fail the operation and rollback the transaction
		OE_Abort = 2,   // Back out changes but do no rollback transaction
		OE_Fail = 3,   // Stop the operation but leave all prior changes
		OE_Ignore = 4,   // Ignore the error. Do not do the INSERT or UPDATE
		OE_Replace = 5,   // Delete existing record, then do INSERT or UPDATE
		OE_Restrict = 6,   // OE_Abort for IMMEDIATE, OE_Rollback for DEFERRED
		OE_SetNull = 7,   // Set the foreign key value to NULL
		OE_SetDflt = 8,   // Set the foreign key value to its default
		OE_Cascade = 9,   // Cascade the changes
		OE_Default = 99,  // Do whatever the default action is
	};

	enum CONFLICT : uint8
	{
		CONFLICT_ROLLBACK = 1,
		CONFLICT_IGNORE = 2,
		CONFLICT_FAIL = 3,
		CONFLICT_ABORT = 4,
		CONFLICT_REPLACE = 5
	};

	enum TYPE : uint8
	{
		TYPE_INTEGER = 1,
		TYPE_FLOAT = 2,
		TYPE_BLOB = 4,
		TYPE_NULL = 5,
		TYPE_TEXT = 3,
	};

	struct IndexSample;
	struct Index
	{
		char *Name;					// Name of this index
		array_t2<uint16, int> Columns;   // Which columns are used by this index.  1st is 0 // Number of columns in table used by this index
		tRowcnt *RowEsts;			// From ANALYZE: Est. rows selected by each column
		Table *Table;				// The SQL table being indexed
		char *ColAff;				// String defining the affinity of each column
		Index *Next;				// The next index associated with the same table
		Schema *Schema;				// Schema containing this index
		SO *SortOrders;				// for each column: True==DESC, False==ASC
		char **CollNames;			// Array of collation sequence names for index
		int Id;						// DB Page containing root of this index
		OE OnError;					// OE_Abort, OE_Ignore, OE_Replace, or OE_None
		unsigned AutoIndex:2;		// 1==UNIQUE, 2==PRIMARY KEY, 0==CREATE INDEX
		unsigned Unordered:1;		// Use this index for == or IN queries only
#ifdef ENABLE_STAT3		  
		tRowcnt AvgEq;					// Average nEq value for key values not in aSample
		array_t<IndexSample> Samples;   // Samples of the left-most key
#endif
	};

	struct IndexSample
	{
		union
		{
			char *Z;				// Value if eType is SQLITE_TEXT or SQLITE_BLOB */
			double R;				// Value if eType is SQLITE_FLOAT */
			int64 I;				// Value if eType is SQLITE_INTEGER */
		} u;
		TYPE Type;					// SQLITE_NULL, SQLITE_INTEGER ... etc.
		int Bytes;					// Size in byte of text or blob.
		tRowcnt Eqs;				// Est. number of rows where the key equals this sample
		tRowcnt Lts;				// Est. number of rows where key is less than this sample
		tRowcnt DLts;				// Est. number of distinct keys less than this sample
	};

	struct Token
	{
		const char *data;			// Text of the token.  Not NULL-terminated!
		uint32 length;				// Number of characters in this token
	};

	struct Expr;
	struct ExprList;
	struct AggInfo
	{
		struct AggInfoColumn
		{
			Table *Table;           // Source table
			int TableID;            // Cursor number of the source table
			int Column;				// Column number within the source table
			int SorterColumn;		// Column number in the sorting index
			int Mem;				// Memory location that acts as accumulator
			Expr *Expr;				// The original expression
		};
		struct AggInfoFunc
		{   
			Expr *Expr;             // Expression encoding the function
			FuncDef *Func;          // The aggregate function implementation
			int Mem;                // Memory location that acts as accumulator
			int Distinct;           // Ephemeral table used to enforce DISTINCT
		};
		uint8 DirectMode;			// Direct rendering mode means take data directly from source tables rather than from accumulators
		uint8 UseSortingIdx;		// In direct mode, reference the sorting index rather than the source table
		int SortingIdx;				// Cursor number of the sorting index
		int SortingIdxPTab;			// Cursor number of pseudo-table
		int SortingColumns;			// Number of columns in the sorting index
		ExprList *GroupBy;			// The group by clause
		array_t<AggInfoColumn> Columns; // For each column used in source tables
		int Accumulators;			// Number of columns that show through to the output. Additional columns are used only as parameters to aggregate functions
		array_t<AggInfoFunc> Funcs; // For each aggregate function

		//? make Columns -> Cols
	};

	enum AFF : uint8
	{
		AFF_TEXT = 'a',
		AFF_NONE = 'b',
		AFF_NUMERIC = 'c',
		AFF_INTEGER = 'd',
		AFF_REAL = 'e',
		AFF_MASK = 0x67, // The SQLITE_AFF_MASK values masks off the significant bits of an affinity value. 
		// Additional bit values that can be ORed with an affinity without changing the affinity.
		AFF_BIT_JUMPIFNULL = 0x08, // jumps if either operand is NULL
		AFF_BIT_STOREP2 = 0x10,	// Store result in reg[P2] rather than jump
		AFF_BIT_NULLEQ = 0x80,  // NULL=NULL
	};
	__device__ inline void operator|=(AFF &a, int b) { a = (AFF)(a | b); }
	__device__ inline AFF operator|(AFF a, AFF b) { return (AFF)((int)a | (int)b); }

#define IsNumericAffinity(X) ((X) >= AFF_NUMERIC)

#pragma endregion

#pragma region Select

	struct IdList
	{
		struct IdListItem
		{
			char *Name;			// Name of the identifier
			int Idx;			// Index in some Table.aCol[] of a column named zName
		};
		array_t<IdListItem> Ids;
	};

	enum JT : uint8
	{
		JT_INNER = 0x0001,		// Any kind of inner or cross join
		JT_CROSS = 0x0002,		// Explicit use of the CROSS keyword
		JT_NATURAL = 0x0004,	// True for a "natural" join
		JT_LEFT = 0x0008,		// Left outer join
		JT_RIGHT = 0x0010,		// Right outer join
		JT_OUTER = 0x0020,		// The "OUTER" keyword is present
		JT_ERROR = 0x0040,		// unknown or unsupported join type
	};
	__device__ inline void operator|=(JT &a, JT b) { a = (JT)(a | b); }
	__device__ inline JT operator|(JT a, JT b) { return (JT)((int)a | (int)b); }

	struct Select;
	struct SrcList
	{
		struct SrcListItem
		{
			Schema *Schema;		// Schema to which this item is fixed
			char *Database;		// Name of database holding this table
			char *Name;			// Name of the table
			char *Alias;		// The "B" part of a "A AS B" phrase.  zName is the "A"
			Table *Table;		// An SQL table corresponding to zName
			Select *Select;		// A SELECT statement used in place of a table name
			int AddrFillSub;	// Address of subroutine to manifest a subquery
			int RegReturn;		// Register holding return address of addrFillSub
			JT Jointype;		// Type of join between this able and the previous
			unsigned NotIndexed:1;    // True if there is a NOT INDEXED clause
			unsigned IsCorrelated:1;  // True if sub-query is correlated
			unsigned ViaCoroutine:1;  // Implemented as a co-routine
#ifndef OMIT_EXPLAIN
			uint8 SelectId;     // If pSelect!=0, the id of the sub-select in EQP
#endif
			int Cursor;			// The VDBE cursor number used to access this table
			Expr *On;			// The ON clause of a join
			IdList *Using;		// The USING clause of a join
			Bitmask ColUsed;	// Bit N (1<<N) set if column N of pTab is used
			char *IndexName;    // Identifier from "INDEXED BY <zIndex>" clause
			Index *Index;		// Index structure corresponding to zIndex, if any
		};
		int16 Srcs;				// Number of tables or subqueries in the FROM clause
		int16 Allocs;			// Number of entries allocated in a[] below
		SrcListItem Ids[1];		// One entry for each identifier on the list
	};

	enum WHERE : uint16
	{
		WHERE_ORDERBY_NORMAL = 0x0000,	// No-op
		WHERE_ORDERBY_MIN = 0x0001,		// ORDER BY processing for min() func
		WHERE_ORDERBY_MAX = 0x0002,		// ORDER BY processing for max() func
		WHERE_ONEPASS_DESIRED = 0x0004, // Want to do one-pass UPDATE/DELETE
		WHERE_DUPLICATES_OK = 0x0008,	// Ok to return a row more than once
		WHERE_OMIT_OPEN_CLOSE = 0x0010, // Table cursors are already open
		WHERE_FORCE_TABLE = 0x0020,		// Do not use an index-only search
		WHERE_ONETABLE_ONLY = 0x0040,	// Only code the 1st table in pTabList
		WHERE_AND_ONLY = 0x0080,		// Don't use indices for OR terms
	};
	__device__ inline void operator|=(WHERE &a, int b) { a = (WHERE)(a | b); }
	__device__ inline WHERE operator|(WHERE a, WHERE b) { return (WHERE)((int)a | (int)b); }

	struct WhereTerm;
	struct WherePlan
	{
		uint32 WsFlags;					// WHERE_* flags that describe the strategy
		uint16 Eqs;                     // Number of == constraints
		uint16 OBSats;                  // Number of ORDER BY terms satisfied
		double Rows;					// Estimated number of rows (for EQP)
		union
		{
			Index *Index;               // Index when WHERE_INDEXED is true
			WhereTerm *Term;			// WHERE clause term for OR-search
			IIndexInfo *VTableIndex;	// Virtual table index to use
		} u;
	};

	struct WhereLevel
	{
		struct InLoop
		{
			int Cur;			// The VDBE cursor used by this IN operator
			int AddrInTop;		// Top of the IN loop
			uint8 EndLoopOp;	// IN Loop terminator. OP_Next or OP_Prev
		};
		WherePlan Plan;			// query plan for this element of the FROM clause
		int LeftJoin;			// Memory cell used to implement LEFT OUTER JOIN
		int TabCur;				// The VDBE cursor used to access the table
		int IdxCur;				// The VDBE cursor used to access pIdx
		int AddrBrk;			// Jump here to break out of the loop
		int AddrNxt;			// Jump here to start the next IN combination
		int AddrCont;			// Jump here to continue with the next loop cycle
		int AddrFirst;			// First instruction of interior of the loop
		uint8 From;				// Which entry in the FROM clause
		uint8 OP, P5;           // Opcode and P5 of the opcode that ends the loop
		int P1, P2;				// Operands of the opcode used to ends the loop
		union
		{
			struct
			{
				InLoop *InLoops; // Information about each nested IN operator
				int InLoopsLength;
			} in;						// Used when plan.wsFlags&WHERE_IN_ABLE
			Index *Covidx;				// Possible covering index for WHERE_MULTI_OR
		} u;							// Information that depends on plan.wsFlags
		double OptCost;					// "Optimal" cost for this level
		// The following field is really not part of the current level.  But we need a place to cache virtual table index information for each
		// virtual table in the FROM clause and the WhereLevel structure is a convenient place since there is one WhereLevel for each FROM clause element.
		IIndexInfo *IndexInfo;			// Index info for n-th source table
	};

	enum WHERE_DISTINCT : uint8
	{
		WHERE_DISTINCT_NOOP = 0,		// DISTINCT keyword not used
		WHERE_DISTINCT_UNIQUE = 1,		// No duplicates
		WHERE_DISTINCT_ORDERED = 2,		// All duplicates are adjacent
		WHERE_DISTINCT_UNORDERED = 3,	// Duplicates are scattered
	};

	struct Parse;
	struct WhereClause;
	struct WhereInfo
	{
		Parse *Parse;				// Parsing and code generating context
		SrcList *TabList;			// List of tables in the join
		uint16 OBSats;              // Number of ORDER BY terms satisfied by indices
		WHERE WctrlFlags;           // Flags originally passed to sqlite3WhereBegin()
		bool OkOnePass;				// Ok to use one-pass algorithm for UPDATE/DELETE
		uint8 UntestedTerms;        // Not all WHERE terms resolved by outer loop
		WHERE_DISTINCT EDistinct;   // One of the WHERE_DISTINCT_* values below
		int TopId;					// The very beginning of the WHERE loop
		int ContinueId;				// Jump here to continue with next record
		int BreakId;				// Jump here to break out of the loop
		int Levels;					// Number of nested loop
		WhereClause *WC;			// Decomposition of the WHERE clause
		double SavedNQueryLoop;		// pParse->nQueryLoop outside the WHERE loop
		double RowOuts;				// Estimated number of output rows
		WhereLevel Data[1];			// Information about each nest loop in WHERE
		__device__ static WhereInfo *Begin(::Parse *parse, SrcList *tabList, Expr *where,  ExprList *orderBy, ExprList *distinct, WHERE wctrlFlags, int idxCur);
		__device__ static void End(WhereInfo *winfo);
	};

	enum NC : uint8
	{
		NC_AllowAgg = 0x01,			// Aggregate functions are allowed here
		NC_HasAgg = 0x02,			// One or more aggregate functions seen
		NC_IsCheck = 0x04,			// True if resolving names in a CHECK constraint
		NC_InAggFunc = 0x08,		// True if analyzing arguments to an agg func
	};
	__device__ inline void operator|=(NC &a, int b) { a = (NC)(a | b); }
	__device__ inline void operator&=(NC &a, int b) { a = (NC)(a & b); }
	__device__ inline NC operator~(NC a) { return (NC)(~(int)a); }

	struct NameContext
	{
		Parse *Parse;				// The parser
		SrcList *SrcList;			// One or more tables used to resolve names
		ExprList *EList;			// Optional list of named expressions
		AggInfo *AggInfo;			// Information about aggregates at this level
		NameContext *Next;			// Next outer name context.  NULL for outermost
		int Refs;					// Number of names resolved by this context
		int Errs;					// Number of errors encountered while resolving names
		NC NCFlags;					// Zero or more NC_* flags defined below
	};

	enum SF : uint16
	{
		SF_Distinct = 0x0001,		// Output should be DISTINCT
		SF_Resolved = 0x0002,		// Identifiers have been resolved
		SF_Aggregate = 0x0004,		// Contains aggregate functions
		SF_UsesEphemeral = 0x0008,  // Uses the OpenEphemeral opcode
		SF_Expanded = 0x0010,		// sqlite3SelectExpand() called on this
		SF_HasTypeInfo = 0x0020,	// FROM subqueries have Table metadata
		SF_UseSorter = 0x0040,		// Sort using a sorter
		SF_Values = 0x0080,			// Synthesized from VALUES clause
		SF_Materialize = 0x0100,	// Force materialization of views
		SF_NestedFrom = 0x0200,		// Part of a parenthesized FROM clause
	};
	__device__ inline void operator|=(SF &a, SF b) { a = (SF)(a | b); }
	__device__ inline void operator&=(SF &a, SF b) { a = (SF)(a & b); }
	__device__ inline SF operator|(SF a, SF b) { return (SF)((int)a | (int)b); }
	__device__ inline SF operator&(SF a, SF b) { return (SF)((int)a & (int)b); }
	__device__ inline SF operator~(SF a) { return (SF)(~(int)a); }

	enum SRT : uint8
	{
		SRT_Union = 1,				// Store result as keys in an index
		SRT_Except = 2,				// Remove result from a UNION index
		SRT_Exists = 3,				// Store 1 if the result is not empty
		SRT_Discard = 4,			// Do not save the results anywhere
		// IgnorableOrderby(x) : The ORDER BY clause is ignored for all of the above
		SRT_Output = 5,				// Output each row of result
		SRT_Mem = 6,				// Store result in a memory cell
		SRT_Set = 7,				// Store results as keys in an index
		SRT_Table = 8,				// Store result as data with an automatic rowid
		SRT_EphemTab = 9,			// Create transient tab and store like SRT_Table
		SRT_Coroutine = 10,			// Generate a single row of result
	};

	struct SelectDest
	{
		SRT Dest;			// How to dispose of the results.  On of SRT_* above.
		AFF AffSdst;		// Affinity used when eDest==SRT_Set
		int SDParmId;		// A parameter used by the eDest disposal method
		int SdstId;			// Base register where results are written
		int Sdsts;			// Number of registers allocated
	};

	class Vdbe;
	struct Select
	{
		ExprList *EList;			// The fields of the result
		TK OP;						// One of: TK_UNION TK_ALL TK_INTERSECT TK_EXCEPT
		SF SelFlags;				// Various SF_* values
		int LimitId, OffsetId;		// Memory registers holding LIMIT & OFFSET counters
		::OP AddrOpenEphms[3];		// OP_OpenEphem opcodes related to this select
		double SelectRows;			// Estimated number of result rows
		SrcList *Src;				// The FROM clause
		Expr *Where;				// The WHERE clause
		ExprList *GroupBy;			// The GROUP BY clause
		Expr *Having;				// The HAVING clause
		ExprList *OrderBy;			// The ORDER BY clause
		Select *Prior;				// Prior select in a compound select statement
		Select *Next;				// Next select to the left in a compound
		Select *Rightmost;			// Right-most select in a compound select statement
		Expr *Limit;				// LIMIT expression. NULL means not used.
		Expr *Offset;				// OFFSET expression. NULL means not used.

		__device__ static void DestInit(SelectDest *dest, SRT dest2, int parmId);
		__device__ static Select *New(Parse *parse, ExprList *list, SrcList *src, Expr *where, ExprList *groupBy, Expr *having, ExprList *orderBy, SF selFlags, Expr *limit, Expr *offset);
		__device__ static void Delete(Context *ctx, Select *p);
		__device__ static JT JoinType(Parse *parse, Token *a, Token *b, Token *c);
		__device__ bool ProcessJoin(Parse *parse);
		__device__ Table *ResultSetOfSelect(Parse *parse);
		//__device__ static Vdbe *GetVdbe(Parse *parse);
		__device__ static RC IndexedByLookup(Parse *parse, SrcList::SrcListItem *from);
		__device__ void Expand(Parse *parse);
		__device__ void AddTypeInfo(Parse *parse);
		__device__ void Prep(Parse *parse, NameContext *outerNC);
		__device__ static RC Select_(Parse *parse, Select *p, SelectDest *dest);
		__device__ static void ExplainSelect(Vdbe *v, Select *p);
	};

#define IgnorableOrderby(x) ((x->Dest) <= SRT_Discard)

#pragma endregion

#pragma region Expr

	enum EP : uint16
	{
		EP_FromJoin = 0x0001,		// Originated in ON or USING clause of a join
		EP_Agg = 0x0002,			// Contains one or more aggregate functions
		EP_Resolved = 0x0004,		// IDs have been resolved to COLUMNs
		EP_Error = 0x0008,			// Expression contains one or more errors
		EP_Distinct = 0x0010,		// Aggregate function with DISTINCT keyword
		EP_VarSelect = 0x0020,		// pSelect is correlated, not constant
		EP_DblQuoted = 0x0040,		// token.z was originally in "..."
		EP_InfixFunc = 0x0080,		// True for an infix function: LIKE, GLOB, etc
		EP_Collate = 0x0100,		// Tree contains a TK_COLLATE opeartor
		EP_FixedDest = 0x0200,		// Result needed in a specific register
		EP_IntValue = 0x0400,		// Integer value contained in u.iValue
		EP_xIsSelect = 0x0800,		// x.pSelect is valid (otherwise x.pList is)
		EP_Hint = 0x1000,			// Not used
		EP_Reduced = 0x2000,		// Expr struct is EXPR_REDUCEDSIZE bytes only
		EP_TokenOnly = 0x4000,		// Expr struct is EXPR_TOKENONLYSIZE bytes only
		EP_Static = 0x8000,			// Held in memory not obtained from malloc()
	};
	__device__ inline void operator|=(EP &a, int b) { a = (EP)(a | b); }
	__device__ inline void operator&=(EP &a, int b) { a = (EP)(a & b); }
	__device__ inline EP operator|(EP a, EP b) { return (EP)((int)a | (int)b); }
	__device__ inline EP operator&(EP a, EP b) { return (EP)((int)a & (int)b); }

	enum EP2 : uint8
	{
		EP2_MallocedToken = 0x0001,	// Need to sqlite3DbFree() Expr.zToken
		EP2_Irreducible = 0x0002,	// Cannot EXPRDUP_REDUCE this Expr
	};
	__device__ inline void operator|=(EP2 &a, int b) { a = (EP2)(a | b); }

	__constant__ extern const Token g_intTokens[];

	class Vdbe;
	struct ExprSpan;
	struct ExprList;
	enum IN_INDEX : uint8;
	struct Expr
	{
		TK OP;					// Operation performed by this node
		AFF Aff;					// The affinity of the column or 0 if not a column
		EP Flags;					// Various flags.  EP_* See below
		union
		{
			char *Token;			// Token value. Zero terminated and dequoted
			int I;					// Non-negative integer value if EP_IntValue
		} u;

		// If the EP_TokenOnly flag is set in the Expr.flags mask, then no space is allocated for the fields below this point. An attempt to
		// access them will result in a segfault or malfunction. 
		Expr *Left;					// Left subnode
		Expr *Right;				// Right subnode
		union
		{
			ExprList *List;			// Function arguments or in "<expr> IN (<expr-list)"
			Select *Select;			// Used for sub-selects and "<expr> IN (<select>)"
		} x;

		// If the EP_Reduced flag is set in the Expr.flags mask, then no space is allocated for the fields below this point. An attempt to
		// access them will result in a segfault or malfunction.
#if MAX_EXPR_DEPTH > 0
		int Height;					// Height of the tree headed by this node
#endif
		// TK_COLUMN: cursor number of table holding column
		// TK_REGISTER: register number
		// TK_TRIGGER: 1 -> new, 0 -> old
		int TableId;            
		// TK_COLUMN: column index.  -1 for rowid.
		// TK_VARIABLE: variable number (always >= 1).
		yVars ColumnIdx;
		int16 Agg;					// Which entry in pAggInfo->aCol[] or ->aFunc[]
		int16 RightJoinTable;		// If EP_FromJoin, the right table of the join
		EP2 Flags2;					// Second set of flags.  EP2_...
		// TK_REGISTER: original value of Expr.op
		// TK_COLUMN: the value of p5 for OP_Column
		// TK_AGG_FUNCTION: nesting depth
		uint8 OP2;                
		// Used by TK_AGG_COLUMN and TK_AGG_FUNCTION
		AggInfo *AggInfo;
		// Table for TK_COLUMN expressions.
		Table *Table;

		__device__ AFF Affinity();
		__device__ static Expr *AddCollateToken(Parse *parse, Expr *expr, Token *collName);
		__device__ static Expr *AddCollateString(Parse *parse, Expr *expr, const char *z);
		__device__ Expr *SkipCollate();
		__device__ CollSeq *CollSeq(Parse *parse);
		__device__ AFF CompareAffinity(AFF aff2);
		__device__ bool ValidIndexAffinity(AFF indexAff);
		__device__ static Core::CollSeq *BinaryCompareCollSeq(Parse *parse, Expr *left, Expr *right);
#if MAX_EXPR_DEPTH > 0
		__device__ static RC CheckHeight(Parse *parse, int height);
		__device__ void SetHeight(Parse *parse);
		__device__ static int SelectExprHeight(Select *select);
#endif
		__device__ static Expr *Alloc(Context *ctx, int op, const Token *token, bool dequote);
		__device__ static Expr *Expr_(Context *ctx, int op, const char *token);
		__device__ static void AttachSubtrees(Context *ctx, Expr *root, Expr *left, Expr *right);
		__device__ static Expr *PExpr_(Parse *parse, int op, Expr *left, Expr *right, const Token *token);
		__device__ static Expr *And(Context *ctx, Expr *left, Expr *right);
		__device__ static Expr *Function(Parse *parse, ExprList *list, Token *token);
		__device__ static void AssignVarNumber(Parse *parse, Expr *expr);
		__device__ static void Delete(Context *ctx, Expr *expr);
		__device__ static Expr *Dup(Context *ctx, Expr *expr, int flags);
		__device__ static ExprList *ListDup(Context *ctx, ExprList *list, int flags);
#if !defined(OMIT_VIEW) || !defined(OMIT_TRIGGER) || !defined(OMIT_SUBQUERY)
		__device__ static SrcList *SrcListDup(Context *ctx, SrcList *list, int flags);
		__device__ static IdList *IdListDup(Context *ctx, IdList *list);
#endif
		__device__ static Select *SelectDup(Context *ctx, Select *select, int flags);

		__device__ static ExprList *ListAppend(Parse *parse, ExprList *list, Expr *expr);
		__device__ static void ListSetName(Parse *parse, ExprList *list, Token *name, bool dequote);
		__device__ static void ListSetSpan(Parse *parse, ExprList *list, ExprSpan *span);
		__device__ static void ListCheckLength(Parse *parse, ExprList *lList, const char *object);
		__device__ static void ListDelete(Context *ctx, ExprList *list);
		__device__ bool IsConstant();
		__device__ bool IsConstantNotJoin();
		__device__ bool IsConstantOrFunction();
		__device__ bool IsInteger(int *value);
		__device__ bool CanBeNull();
		__device__ static void CodeIsNullJump(Vdbe *v, const Expr *expr, int reg, int dest);
		__device__ bool NeedsNoAffinityChange(AFF aff);
		__device__ static bool IsRowid(const char *z);
		__device__ static int CodeOnce(Parse *parse);
#ifndef OMIT_SUBQUERY
		__device__ static IN_INDEX FindInIndex(Parse *parse, Expr *expr, int *notFound);
		__device__ static int CodeSubselect(Parse *parse, Expr *expr, int mayHaveNull, bool isRowid);
		__device__ static void CodeIN(Parse *parse, Expr *expr, int destIfFalse, int destIfNull);
#endif
		__device__ static void CacheStore(Parse *parse, int table, int column, int reg);
		__device__ static void CacheRemove(Parse *parse, int reg, int regs);
		__device__ static void CachePush(Parse *parse);
		__device__ static void CachePop(Parse *parse, int n);
		__device__ static void CachePinRegister(Parse *parse, int reg);
		__device__ static void CodeGetColumnOfTable(Vdbe *v, Core::Table *table, int tabCur, int column, int regOut);
		__device__ static int CodeGetColumn(Parse *parse, Core::Table *table, int column, int tableId, int reg, uint8 p5);
		__device__ static void CacheClear(Parse *parse);
		__device__ static void CacheAffinityChange(Parse *parse, int start, int count);
		__device__ static void CodeMove(Parse *parse, int from, int to, int regs);
		__device__ static int CodeTarget(Parse *parse, Expr *expr, int target);
		__device__ static int CodeTemp(Parse *parse, Expr *expr, int *reg);
		__device__ static int Code(Parse *parse, Expr *expr, int target);
		__device__ static int CodeAndCache(Parse *parse, Expr *expr, int target);
#ifdef ENABLE_TREE_EXPLAIN
		__device__ static void ExplainExpr(Vdbe *o, Expr *expr);
		__device__ static void ExplainExprList(Vdbe *v, ExprList *list);
#endif
		__device__ static void CodeConstants(Parse *parse, Expr *expr);
		__device__ static int CodeExprList(Parse *parse, ExprList *list, int target, bool doHardCopy);
		__device__ void IfTrue(Parse *parse, int dest, AFF jumpIfNull);
		__device__ void IfFalse(Parse *parse, int dest, AFF jumpIfNull);
		__device__ static int Compare(Expr *a, Expr *b);
		__device__ static int ListCompare(ExprList *a, ExprList *b);
		__device__ bool FunctionUsesThisSrc(SrcList *srcList);
		__device__ static void AnalyzeAggregates(NameContext *nc, Expr *expr);
		__device__ static void AnalyzeAggList(NameContext *nc, ExprList *list);
		__device__ static int GetTempReg(Parse *parse);
		__device__ static void ReleaseTempReg(Parse *parse, int reg);
		__device__ static int GetTempRange(Parse *parse, int regs);
		__device__ static void ReleaseTempRange(Parse *parse, int reg, int regs);
		__device__ static void ClearTempRegCache(Parse *parse);
	};

#ifdef _DEBUG
#define ExprSetIrreducible(x) (x)->Flags2 |= EP2_Irreducible
#else
#define ExprSetIrreducible(x)
#endif

#define ExprHasProperty(x,p)     (((x)->Flags&(p))==(p))
#define ExprHasAnyProperty(x,p)  (((x)->Flags&(p))!=0)
#define ExprSetProperty(x,p)     (x)->Flags|=(p)
#define ExprClearProperty(x,p)   (x)->Flags&=~(p)

#define EXPR_FULLSIZE           sizeof(Expr)				// Full size
#define EXPR_REDUCEDSIZE        offsetof(Expr, TableId)	// Common features
#define EXPR_TOKENONLYSIZE      offsetof(Expr, Left)		// Fewer features
#define EXPRDUP_REDUCE         0x0001  // Used reduced-size Expr nodes

	struct ExprList
	{
		struct ExprListItem
		{
			Expr *Expr;				// The list of expressions
			char *Name;				// Token associated with this expression
			char *Span;				// Original text of the expression
			SO SortOrder;			// 1 for DESC or 0 for ASC
			unsigned Done:1;		// A flag to indicate when processing is finished
			unsigned SpanIsTab:1;	// zSpan holds DB.TABLE.COLUMN
			uint16 OrderByCol;      // For ORDER BY, column number in result set
			uint16 Alias;           // Index into Parse.aAlias[] for zName
		};
		int Exprs;					// Number of expressions on the list
		int ECursor;				// VDBE Cursor associated with this ExprList
		// For each expression in the list
		ExprListItem *Ids;			// Alloc a power of two greater or equal to nExpr
	};

	struct ExprSpan
	{
		Expr *Expr;          // The expression parse tree
		const char *Start;   // First character of input text
		const char *End;     // One character past the end of input text
	};

#pragma endregion

#pragma region Walker

	enum WRC : uint8
	{
		WRC_Continue = 0,	// Continue down into children
		WRC_Prune = 1,		// Omit children but continue walking siblings
		WRC_Abort = 2,		// Abandon the tree walk
	};

	struct SrcCount;
	struct Walker
	{
		WRC (*ExprCallback)(Walker *w, Expr *expr);			// Callback for expressions
		WRC (*SelectCallback)(Walker *w, Select *select);	// Callback for SELECTs
		Parse *Parse;                   // Parser context.
		int WalkerDepth;				// Number of subqueries
		union
		{
			NameContext *NC;            // Naming context
			int I;                      // Integer value
			SrcList *SrcList;           // FROM clause
			SrcCount *SrcCount;			// Counting column references
		} u; // Extra data for callback

#pragma region From: Resolve_c
		__device__ static bool MatchSpanName(const char *span, const char *colName, const char *table, const char *dbName);
		__device__ static Expr *CreateColumnExpr(Context *ctx, SrcList *src, int srcId, int colId);
		__device__ static bool ResolveOrderGroupBy(Core::Parse *parse, Select *select, ExprList *orderBy, const char *type);
		__device__ static bool ResolveExprNames(NameContext *nc, Expr *expr);
		__device__ static void ResolveSelectNames(Core::Parse *parse, Select *p, NameContext *outerNC);
#pragma endregion

#pragma region From: Walker_c
		__device__ WRC WalkExpr(Expr *expr);
		__device__ WRC WalkExprList(ExprList *list);
		__device__ WRC WalkSelect(Select *Select);
		__device__ WRC WalkSelectExpr(Select *left);
		__device__ WRC WalkSelectFrom(Select *left);
#pragma endregion
	};

#pragma endregion

#pragma region Callback

	struct Callback
	{
		__device__ static CollSeq *GetCollSeq(Parse *parse, TEXTENCODE encode, CollSeq *coll, const char *name);
		__device__ static RC CheckCollSeq(Parse *parse, CollSeq *coll);
		__device__ static CollSeq *FindCollSeq(Context *ctx, TEXTENCODE encode, const char *name, bool create);
		__device__ static void FuncDefInsert(FuncDefHash *hash, FuncDef *def);
		__device__ static FuncDef *FindFunction(Context *ctx, const char *name, int nameLength, int args, TEXTENCODE encode, bool createFlag);
		__device__ static void SchemaClear(void *p);
		__device__ static Schema *SchemaGet(Context *ctx, Btree *bt);
	};

#pragma endregion

#pragma region Prepare

	typedef struct
	{
		Context *Ctx; // The database being initialized
		char **ErrMsg; // Error message stored here
		int Db; // 0 for main database.  1 for TEMP, 2.. for ATTACHed
		RC RC; // Result code stored here
	} InitData;

	struct Prepare
	{
		__device__ static bool InitCallback(void *init, int argc, char **argv, char **notUsed1);
		__device__ static RC InitOne(Context *ctx, int db, char **errMsg);
		__device__ static RC Init(Context *ctx, char **errMsg);
		__device__ static RC ReadSchema(Parse *parse);
		__device__ static int SchemaToIndex(Context *ctx, Schema *schema);
		__device__ static RC Prepare_(Context *ctx, const char *sql, int bytes, bool isPrepareV2, Vdbe *reprepare, Vdbe **stmtOut, const char **tailOut);
		__device__ static RC LockAndPrepare(Context *ctx, const char *sql, int bytes, bool isPrepareV2, Vdbe *reprepare, Vdbe **stmtOut, const char **tailOut);
		__device__ static RC Reprepare(Vdbe *p);
		__device__ static RC Prepare_(Context *ctx, const char *sql, int bytes, Vdbe **stmtOut, const char **tailOut);
		__device__ static RC Prepare_v2(Context *ctx, const char *sql, int bytes, Vdbe **stmtOut, const char **tailOut);
#ifndef OMIT_UTF16
		__device__ static RC Prepare16(Context *ctx, const void *sql, int bytes, bool isPrepareV2, Vdbe **stmtOut, const void **tailOut);
		__device__ static RC Prepare16(Context *ctx, const void *sql, int bytes, Vdbe **stmtOut, const void **tailOut);
		__device__ static RC Prepare16_v2(Context *ctx, const void *sql, int bytes, Vdbe **stmtOut, const void **tailOut);
#endif
	};

#pragma endregion 

#pragma region Parse

#ifdef OMIT_VIRTUALTABLE
#define INDECLARE_VTABLE(x) false
#else
#define INDECLARE_VTABLE(x) (x->DeclareVTable)
#endif

	struct AutoincInfo
	{
		AutoincInfo *Next;		// Next info block in a list of them all
		Table *Table;			// Table this info block refers to
		int DB;					// Index in sqlite3.aDb[] of database holding pTab
		int RegCtr;				// Memory register holding the rowid counter
	};

	enum IN_INDEX : uint8
	{
		IN_INDEX_ROWID = 1,
		IN_INDEX_EPH = 2,
		IN_INDEX_INDEX_ASC = 3,
		IN_INDEX_INDEX_DESC = 4,
	};

#ifndef OMIT_TRIGGER
#define Parse_Toplevel(p) p
#else
#define Parse_Toplevel(p) ((p)->Toplevel ? (p)->Toplevel : (p))
#endif

	struct FKey;
	struct TableLock;
	struct Trigger;
	struct TriggerPrg;
	struct Parse
	{
		struct ColCache
		{
			int Table;				// Table cursor number
			int Column;				// Table column number
			uint8 TempReg;			// iReg is a temp register that needs to be freed
			int Level;				// Nesting level
			int Reg;				// Reg with value of this column. 0 means none.
			int Lru;				// Least recently used entry has the smallest value
		};

		Context *Ctx;				// The main database structure
		char *ErrMsg;				// An error message
		Vdbe *V;					// An engine for executing database bytecode
		RC RC;						// Return code from execution
		uint8 ColNamesSet;			// TRUE after OP_ColumnName has been issued to pVdbe
		uint8 CheckSchema;			// Causes schema cookie check after an error
		uint8 Nested;				// Number of nested calls to the parser/code generator
		//uint8 TempReg;			// Number of temporary registers in aTempReg[]
		uint8 TempRegsInUse;		// Number of aTempReg[] currently checked out
		//uint8 ColCaches;			// Number of entries in aColCache[]
		uint8 ColCacheIdx;			// Next entry in aColCache[] to replace
		uint8 IsMultiWrite;			// True if statement may modify/insert multiple rows
		bool _MayAbort;				// True if statement may throw an ABORT exception
		array_t3<uint8, int, 8> TempReg; // Holding area for temporary registers
		int RangeRegs;				// Size of the temporary register block
		int RangeRegIdx;			// First register in temporary register block
		int Errs;					// Number of errors seen
		int Tabs;					// Number of previously allocated VDBE cursors
		int Mems;					// Number of memory cells used so far
		int Sets;					// Number of sets used so far
		int Onces;					// Number of OP_Once instructions so far
		int CkBase;					// Base register of data during check constraints
		int CacheLevel;				// ColCache valid when aColCache[].iLevel<=iCacheLevel
		int CacheCnt;				// Counter used to generate aColCache[].lru values
		array_t3<uint8, ColCache, N_COLCACHE> ColCaches; // One for each column cache entry
		yDbMask WriteMask;			// Start a write transaction on these databases
		yDbMask CookieMask;			// Bitmask of schema verified databases
		int CookieGoto;				// Address of OP_Goto to cookie verifier subroutine
		int CookieValue[MAX_ATTACHED + 2];  // Values of cookies to verify
		int RegRowid;				// Register holding rowid of CREATE TABLE entry
		int RegRoot;				// Register holding root page number for new objects
		int MaxArgs;				// Max args passed to user function by sub-program
		Token ConstraintName;		// Name of the constraint currently being parsed
#ifndef OMIT_SHARED_CACHE
		// int TableLocks;			// Number of locks in aTableLock
		array_t<TableLock> TableLocks; // Required table locks for shared-cache mode
#endif
		AutoincInfo *Ainc;			// Information about AUTOINCREMENT counters

		// Information used while coding trigger programs.
		Parse *Toplevel;			// Parse structure for main program (or NULL)
		Table *TriggerTab;			// Table triggers are being coded for
		double QueryLoops;			// Estimated number of iterations of a query
		uint32 Oldmask;				// Mask of old.* columns referenced
		uint32 Newmask;				// Mask of new.* columns referenced
		TK TriggerOP;				// TK_UPDATE, TK_INSERT or TK_DELETE
		OE Orconf;				// Default ON CONFLICT policy for trigger steps
		bool DisableTriggers;		// True to disable triggers

		// Above is constant between recursions.  Below is reset before and after each recursion
		int VarsSeen;				// Number of '?' variables seen in the SQL so far
		//int nzVar;				// Number of available slots in azVar[]
		uint8 Explain;				// True if the EXPLAIN flag is found on the query
#ifndef OMIT_VIRTUALTABLE
		bool DeclareVTable;			// True if inside sqlite3_declare_vtab()
		//int nVtabLock;			// Number of virtual tables to lock
#endif
		//int nAlias;				// Number of aliased result set columns
		int Height;					// Expression tree height of current sub-select
#ifndef OMIT_EXPLAIN
		int SelectId;				// ID of current select for EXPLAIN output
		int NextSelectId;			// Next available select ID for EXPLAIN output
#endif
		array_t<char *>Vars;		// Pointers to names of parameters
		Vdbe *Reprepare;		// VM being reprepared (sqlite3Reprepare())
		array_t<int> Alias;			// Register used to hold aliased result
		const char *Tail;			// All SQL text past the last semicolon parsed
		Table *NewTable;			// A table being constructed by CREATE TABLE
		Trigger *NewTrigger;		// Trigger under construct by a CREATE TRIGGER
		const char *AuthContext;	// The 6th parameter to db->xAuth callbacks
		Token NameToken;			// Token with unqualified schema object name
		Token LastToken;			// The last token parsed
#ifndef OMIT_VIRTUALTABLE
		Token Arg;					// Complete text of a module argument
		array_t<Table *> VTableLocks; // Pointer to virtual tables needing locking
#endif
		Table *ZombieTab;			// List of Table objects to delete after code gen
		TriggerPrg *TriggerPrg;		// Linked list of coded triggers

#pragma region From: Main_c
		__device__ void ErrorMsg(const char *fmt, va_list *args);
#if __CUDACC__
		__device__ inline void ErrorMsg(const char *fmt) { va_list args; va_start(args, nullptr); ErrorMsg(fmt, &args); va_end(args); }
		template <typename T1> __device__ inline void ErrorMsg(const char *fmt, T1 arg1) { va_list args; va_start(args, arg1); ErrorMsg(fmt, &args); va_end(args); }
		template <typename T1, typename T2> __device__ inline void ErrorMsg(const char *fmt, T1 arg1, T2 arg2) { va_list args; va_start(args, arg1, arg2); ErrorMsg(fmt, &args); va_end(args); }
		template <typename T1, typename T2, typename T3> __device__ inline void ErrorMsg(const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list args; va_start(args, arg1, arg2, arg3); ErrorMsg(fmt, &args); va_end(args); }
		template <typename T1, typename T2, typename T3, typename T4> __device__ inline void ErrorMsg(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list args; va_start(args, arg1, arg2, arg3, arg4); ErrorMsg(fmt, &args); va_end(args); }
		template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ inline void ErrorMsg(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5); ErrorMsg(fmt, &args); va_end(args); }
		template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ inline void ErrorMsg(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); ErrorMsg(fmt, &args); va_end(args); }
		template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ inline void ErrorMsg(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); ErrorMsg(fmt, &args); va_end(args); }
		template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ inline void ErrorMsg(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); ErrorMsg(fmt, &args); va_end(args); }
		template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ inline void ErrorMsg(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); ErrorMsg(fmt, &args); va_end(args); }
		template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ inline void ErrorMsg(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); ErrorMsg(fmt, &args); va_end(args); }
#else
		__device__ inline void ErrorMsg(const char *fmt, ...) { va_list args; va_start(args, fmt); ErrorMsg(fmt, &args); va_end(args); }
#endif
#pragma endregion

#pragma region From: Command/Select_c
		__device__ Vdbe *GetVdbe();
#pragma endregion

#pragma region From: Parse+Build_cu
		__device__ void BeginParse(int explainFlag);
#ifndef OMIT_SHARED_CACHE
		__device__ void TableLock(int db, int table, bool isWriteLock, const char *name);
#endif
		__device__ void FinishCoding();
		__device__ void NestedParse(const char *fmt, va_list *args);
#if __CUDACC__
		__device__ inline void NestedParse(const char *fmt) { va_list args; va_start(args, nullptr); NestedParse(fmt, &args); va_end(args); }
		template <typename T1> __device__ inline void NestedParse(const char *fmt, T1 arg1) { va_list args; va_start(args, arg1); NestedParse(fmt, &args); va_end(args); }
		template <typename T1, typename T2> __device__ inline void NestedParse(const char *fmt, T1 arg1, T2 arg2) { va_list args; va_start(args, arg1, arg2); NestedParse(fmt, &args); va_end(args); }
		template <typename T1, typename T2, typename T3> __device__ inline void NestedParse(const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list args; va_start(args, arg1, arg2, arg3); NestedParse(fmt, &args); va_end(args); }
		template <typename T1, typename T2, typename T3, typename T4> __device__ inline void NestedParse(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list args; va_start(args, arg1, arg2, arg3, arg4); NestedParse(fmt, &args); va_end(args); }
		template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ inline void NestedParse(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5); NestedParse(fmt, &args); va_end(args); }
		template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ inline void NestedParse(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); NestedParse(fmt, &args); va_end(args); }
		template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ inline void NestedParse(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); NestedParse(fmt, &args); va_end(args); }
		template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ inline void NestedParse(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); NestedParse(fmt, &args); va_end(args); }
		template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ inline void NestedParse(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); NestedParse(fmt, &args); va_end(args); }
		template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ inline void NestedParse(const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); NestedParse(fmt, &args); va_end(args); }
#else
		__device__ inline void NestedParse(const char *fmt, ...) { va_list args; va_start(args, fmt); NestedParse(fmt, &args); va_end(args); }
#endif
		__device__ static Table *FindTable(Context *ctx, const char *name, const char *dbName);
		__device__ Table *LocateTable(bool isView, const char *name, const char *dbName);
		__device__ Table *LocateTableItem(bool isView,  SrcList::SrcListItem *item);
		__device__ static Index *FindIndex(Context *ctx, const char *name, const char *dbName);
		__device__ static void UnlinkAndDeleteIndex(Context *ctx, int db, const char *indexName);
		__device__ static void CollapseDatabaseArray(Context *ctx);
		__device__ static void ResetOneSchema(Context *ctx, int db);
		__device__ static void ResetAllSchemasOfConnection(Context *ctx);
		__device__ static void CommitInternalChanges(Context *ctx);
		__device__ static void DeleteTable(Context *ctx, Table *table);
		__device__ static void UnlinkAndDeleteTable(Context *ctx, int db, const char *tableName);
		__device__ static char *NameFromToken(Context *ctx, Token *name);
		__device__ void OpenMasterTable(int db);
		__device__ static int FindDbName(Context *ctx, const char *name);
		__device__ static int FindDb(Context *ctx, Token *name);
		__device__ int TwoPartName(Token *name1, Token *name2, Token **unqual);
		__device__ ::RC CheckObjectName(const char *name);
		__device__ void StartTable(Token *name1, Token *name2, bool isTemp, bool isView, bool isVirtual, bool noErr);
		__device__ void AddColumn(Token *name);
		__device__ void AddNotNull(OE onError);
		__device__ static AFF AffinityType(const char *data);
		__device__ void AddColumnType(Token *type);
		__device__ void AddDefaultValue(ExprSpan *span);
		__device__ void AddPrimaryKey(ExprList *list, OE onError, bool autoInc, SO sortOrder);
		__device__ void AddCheckConstraint(Expr *checkExpr);
		__device__ void AddCollateType(Token *token);
		__device__ CollSeq *LocateCollSeq(const char *name);
		__device__ void ChangeCookie(int db);
		__device__ void EndTable(Token *cons, Token *end, Select *select);
#ifndef OMIT_VIEW
		__device__ void CreateView(Token *begin, Token *name1, Token *name2, Select *select, bool isTemp, bool noErr);
#endif
#if !defined(OMIT_VIEW) || !defined(OMIT_VIRTUALTABLE)
		__device__ int ViewGetColumnNames(Table *table);
#endif
#ifndef OMIT_VIEW
		__device__ static void ViewResetAll(Context *ctx, int db);
#endif
#ifndef OMIT_AUTOVACUUM
		__device__ void static RootPageMoved(Context *ctx, int db, int from, int to);
#endif
		__device__ void ClearStatTables(int db, const char *type, const char *name);
		__device__ void CodeDropTable(Table *table, int db, bool isView);
		__device__ void DropTable(SrcList *name, bool isView, bool noErr);
		__device__ void CreateForeignKey(ExprList *fromCol, Token *to, ExprList *toCol, int flags);
		__device__ void DeferForeignKey(bool isDeferred);
		__device__ void RefillIndex(Index *index, int memRootPage);
		__device__ Index *CreateIndex(Token *name1, Token *name2, SrcList *tableName, ExprList *list, OE onError, Token *start, Token *end, int sortOrder, bool ifNotExist);
		__device__ static void DefaultRowEst(Index *index);
		__device__ void DropIndex(SrcList *name, bool ifExists);
		__device__ static void *ArrayAllocate(Context *ctx, void *array_, int entrySize, int *entryLength, int *index);
		__device__ static IdList *IdListAppend(Context *ctx, IdList *list, Token *token);
		__device__ static void IdListDelete(Context *ctx, IdList *list);
		__device__ static int IdListIndex(IdList *list, const char *name);
		__device__ static SrcList *SrcListEnlarge(Context *ctx, SrcList *src, int extra, int start);
		__device__ static SrcList *SrcListAppend(Context *ctx, SrcList *list, Token *table, Token *database);
		__device__ void SrcListAssignCursors(SrcList *list);
		__device__ static void SrcListDelete(Context *ctx, SrcList *list);
		__device__ SrcList *SrcListAppendFromTerm(SrcList *list, Token *table, Token *database, Token *alias, Select *subquery, Expr *on, IdList *using_);
		__device__ void SrcListIndexedBy(SrcList *list, Token *indexedBy);
		__device__ static void SrcListShiftJoinType(SrcList *list);
		__device__ void BeginTransaction(int type);
		__device__ void CommitTransaction();
		__device__ void RollbackTransaction();
		__device__ void Savepoint(IPager::SAVEPOINT op, Token *name);
		__device__ int OpenTempDatabase();
		__device__ void CodeVerifySchema(int db);
		__device__ void CodeVerifyNamedSchema(const char *dbName);
		__device__ void BeginWriteOperation(int setStatement, int db);
		__device__ void MultiWrite();
		__device__ void MayAbort();
		__device__ void HaltConstraint(::RC errCode, int onError, char *p4, int p4type);
		__device__ void Reindex(Token *name1, Token *name2);
		__device__ KeyInfo *IndexKeyinfo(Index *index);
#pragma endregion

#pragma region From: Parse+Complete_cu
		__device__ bool Complete(const char *sql);
#ifndef OMIT_UTF16
		__device__ bool Complete16(const void *sql);
#endif
#pragma endregion

#pragma region From: Parse+FKey_cu

#ifndef OMIT_TRIGGER
		__device__ int FKLocateIndex(Table *parent, FKey *fkey, Index **indexOut, int **colsOut);
		__device__ static FKey *FKReferences(Table *table);
		__device__ void FKDropTable(SrcList *name, Table *table);
		__device__ void FKCheck(Table *table, int regOld, int regNew);
		__device__ uint32 FKOldmask(Table *table);
		__device__ bool FKRequired(Table *table, int *changes, int chngRowid);
		__device__ void FKActions(Table *table, ExprList *changes, int regOld);
#endif
		__device__ static void FKDelete(Context *ctx, Table *table);
#pragma endregion

#pragma region From: Parse+Tokenize_cu
		__device__ static int GetToken(const unsigned char *z, int *tokenType);
		__device__ int RunParser(const char *sql, char **errMsg);
		__device__ static int Dequote(char *z);
#pragma endregion
	};

	// The interface to the LEMON-generated parser
#pragma region From: Parse+Parser_cu
	extern "C" __device__ void *ParserAlloc(void *(*)(size_t));
	extern "C" __device__ void ParserFree(void *, void(*)(void *));
	extern "C" __device__ void Parser(void *, int, Token, Parse *);
#ifdef YYTRACKMAXSTACKDEPTH
	extern "C" __device__ int ParserStackPeak(void *);
#endif
#ifdef _DEBUG
	extern "C" __device__ void ParserTrace(FILE *, char *);
#endif

#pragma endregion

	struct AuthContext
	{
		const char *AuthCtx;		// Put saved Parse.zAuthContext here
		Parse *Parse;				// The Parse structure
	};

#pragma endregion
}
#include "Context.cu.h"
namespace Core {
#pragma region Table

	enum COLFLAG : uint16
	{
		COLFLAG_PRIMKEY = 0x0001,		// Column is part of the primary key
		COLFLAG_HIDDEN = 0x0002,		// A hidden column in a virtual table
	};
	__device__ inline void operator|=(COLFLAG &a, int b) { a = (COLFLAG)(a | b); }

	struct Column
	{
		char *Name;						// Name of this column
		Expr *Dflt;						// Default value of this column
		char *DfltName;					// Original text of the default value
		char *Type;						// Data type for this column
		char *Coll;						// Collating sequence.  If NULL, use the default
		OE NotNull;						// An OE_ code for handling a NOT NULL constraint
		AFF Affinity;					// One of the SQLITE_AFF_... values
		COLFLAG ColFlags;				// Boolean properties.  See COLFLAG_ defines below
	};

	enum VTABLECONFIG : uint8
	{
		VTABLECONFIG_CONSTRAINT = 1,
	};

	struct VTable
	{
		Context *Ctx;				// Database connection associated with this table
		TableModule *Module;		// Pointer to module implementation
		IVTable *IVTable;			// Pointer to vtab instance
		int Refs;					// Number of pointers to this structure
		bool Constraint;			// True if constraints are supported
		int Savepoints;				// Depth of the SAVEPOINT stack
		VTable *Next;				// Next in linked list (see above)

#ifdef OMIT_VIRTUALTABLE
		__device__ inline static void Clear(Context *ctx, Table *table) {}
		__device__ inline static RC Sync(Context *ctx, char **errorOut) { return RC_OK; }
		__device__ static RC Rollback(Context *ctx);
		__device__ static RC Commit(Context *ctx);
		__device__ inline static bool InSync(Context *ctx) { return false; }
		__device__ inline void Lock() {}
		__device__ inline void Unlock() {}
		__device__ inline static void UnlockList(Context *ctx) {}
		__device__ inline static RC Savepoint(Context *ctx, IPager::SAVEPOINT op, int savepoint) { return RC_OK; }
		__device__ inline static VTable *GetVTable(Context *ctx, Table *table) { return nullptr; }
#else
		__device__ static RC CreateModule(Context *ctx, const char *name, const ITableModule *imodule, void *aux, void (*destroy)(void *));
		__device__ void Lock();
		__device__ static VTable *GetVTable(Context *ctx, Table *table);
		__device__ void Unlock();
		__device__ static void Disconnect(Context *ctx, Table *table);
		__device__ static void UnlockList(Context *ctx);
		__device__ static void Clear(Context *ctx, Table *table);
		__device__ static void BeginParse(Parse *parse, Token *name1, Token *name2, Token *moduleName, bool ifNotExists);
		__device__ static void FinishParse(Parse *parse, Token *end);
		__device__ static void ArgInit(Parse *parse);
		__device__ static void ArgExtend(Parse *parse, Token *token);
		__device__ static RC CallConnect(Parse *parse, Table *table);
		__device__ static RC CallCreate(Context *ctx, int dbidx, const char *tableName, char **error);
		__device__ static RC DeclareVTable(Context *ctx, const char *createTableName);
		__device__ static RC CallDestroy(Context *ctx, int db, const char *tableName);
		__device__ static RC Sync(Context *ctx, char **errorOut);
		__device__ static RC Rollback(Context *ctx);
		__device__ static RC Commit(Context *ctx);
		__device__ inline static bool InSync(Context *ctx) { return (ctx->VTrans.length > 0 && ctx->VTrans.data == nullptr); }
		__device__ static RC Begin(Context *ctx, VTable *vtable);
		__device__ static RC Savepoint(Context *ctx, IPager::SAVEPOINT op, int savepoint);
		__device__ static FuncDef *OverloadFunction(Context *ctx, FuncDef *def, int argsLength, Expr *expr);
		__device__ static void MakeWritable(Parse *parse, Table *table);
		__device__ static CONFLICT OnConflict(Context *ctx);
		__device__ static RC Config(Context *ctx, VTABLECONFIG op, void *arg1);
#endif
	};

	enum TF : uint8
	{
		TF_Readonly = 0x01,			// Read-only system table
		TF_Ephemeral = 0x02,		// An ephemeral table
		TF_HasPrimaryKey = 0x04,    // Table has a primary key
		TF_Autoincrement = 0x08,    // Integer primary key is autoincrement
		TF_Virtual = 0x10,			// Is a virtual table
	};
	__device__ inline void operator|=(TF &a, int b) { a = (TF)(a | b); }

	struct Index;
	struct Select;
	struct FKey;
	struct ExprList;
	struct Table
	{
		char *Name;					// Name of the table or view
		array_t2<int16, Column> Cols;   // Information about each column
		Index *Index;				// List of SQL indexes on this table.
		Select *Select;				// NULL for tables.  Points to definition if a view.
		FKey *FKeys;				// Linked list of all foreign keys in this table
		char *ColAff;				// String defining the affinity of each column
#ifndef OMIT_CHECK
		ExprList *Check;			// All CHECK constraints
#endif
		tRowcnt RowEst;				// Estimated rows in table - from sqlite_stat1 table
		int Id;						// Root BTree node for this table (see note above)
		int16 PKey;					// If not negative, use aCol[iPKey] as the primary key
		uint16 Refs;				// Number of pointers to this Table
		TF TabFlags;				// Mask of TF_* values
		OE KeyConf;					// What to do in case of uniqueness conflict on iPKey
#ifndef OMIT_ALTERTABLE
		int AddColOffset;			// Offset in CREATE TABLE stmt to add a new column
#endif
#ifndef OMIT_VIRTUALTABLE
		array_t<char *>ModuleArgs;  // Text of all module args. [0] is module name
		VTable *VTables;			// List of VTable objects.
#endif
		Trigger *Triggers;			// List of triggers stored in pSchema
		Schema *Schema;				// Schema that contains this table
		Table *NextZombie;			// Next on the Parse.pZombieTab list

		__device__ static bool GetTableCallback(void *arg, int columns, char **argv, char **colv);
		__device__ static RC GetTable(Context *db, const char *sql, char ***results, int *rows, int *columns, char **errMsg);
		__device__ static void FreeTable(char **results);
	};

#ifndef OMIT_VIRTUALTABLE
#define IsVirtual(X)      (((X)->TabFlags & TF_Virtual)!=0)
#define IsHiddenColumn(X) (((X)->ColFlags & COLFLAG_HIDDEN)!=0)
#else
#define IsVirtual(X)      0
#define IsHiddenColumn(X) 0
#endif

	struct FKey
	{
		struct ColMap			// Mapping of columns in pFrom to columns in zTo
		{		
			int From;			// Index of column in pFrom
			char *Col;			// Name of column in zTo.  If 0 use PRIMARY KEY
		};
		Table *From;			// Table containing the REFERENCES clause (aka: Child)
		FKey *NextFrom;			// Next foreign key in pFrom
		char *To;				// Name of table that the key points to (aka: Parent)
		FKey *NextTo;			// Next foreign key on table named zTo
		FKey *PrevTo;			// Previous foreign key on table named zTo
		bool IsDeferred;		// True if constraint checking is deferred till COMMIT
		OE Actions[2];       // ON DELETE and ON UPDATE actions, respectively
		Trigger *Triggers[2];	// Triggers for aAction[] actions
		array_t3<int, ColMap, 1> Cols; // One entry for each of nCol column s
	};

#pragma endregion

#pragma region Rowset

	struct RowSet;
	__device__ RowSet *RowSet_Init(Context *ctx, void *space, unsigned int n);
	__device__ void RowSet_Clear(RowSet *p);
	__device__ void RowSet_Insert(RowSet *p, int64 rowid);
	__device__ bool RowSet_Test(RowSet *rowSet, uint8 batch, int64 rowid);
	__device__ bool RowSet_Next(RowSet *p, int64 *rowid);

#pragma endregion

#pragma region Backup

	struct Backup
	{
		Context *DestCtx;        // Destination database handle
		Btree *Dest;            // Destination b-tree file
		uint32 DestSchema;      // Original schema cookie in destination
		bool DestLocked;         // True once a write-transaction is open on pDest
		Pid NextId;				// Page number of the next source page to copy
		Context * SrcCtx;        // Source database handle
		Btree *Src;             // Source b-tree file
		RC RC_;                  // Backup process error code

		// These two variables are set by every call to backup_step(). They are read by calls to backup_remaining() and backup_pagecount().
		Pid Remaining;			// Number of pages left to copy
		Pid Pagecount;			// Total number of pages to copy

		bool IsAttached;		// True once backup has been registered with pager
		Backup *Next;	// Next backup associated with source pager

		__device__ static Backup *Init(Context *destCtx, const char *destDbName, Context *srcCtx, const char *srcDbName);
		__device__ RC Step(int pages);
		__device__ static RC Finish(Backup *p);
		__device__ void Update(Pid page, const uint8 *data);
		__device__ void Restart();
#ifndef OMIT_VACUUM
		__device__ static RC BtreeCopyFile(Btree *to, Btree *from);
#endif
	};

#pragma endregion

#pragma region Internal

	struct Mem;
	struct VdbeCursor;
	struct VdbeFrame;
	struct VdbeFunc;
	struct Explain;

#pragma endregion 

#pragma region Savepoint

	struct Savepoint
	{
		char *Name;				// Savepoint name (nul-terminated)
		int64 DeferredCons;		// Number of deferred fk violations
		Savepoint *Next;		// Parent savepoint (if any)
	};

#pragma endregion
}
#include "Vdbe.cu.h"
namespace Core {
#pragma region Trigger

	enum TRIGGER : uint8
	{
		TRIGGER_BEFORE = 1,
		TRIGGER_AFTER = 2,
	};
	__device__ inline void operator|=(TRIGGER &a, int b) { a = (TRIGGER)(a | b); }
	__device__ inline TRIGGER operator|(TRIGGER a, TRIGGER b) { return (TRIGGER)((int)a | (int)b); }

	struct TriggerStep;
	struct Trigger
	{
		char *Name;				// The name of the trigger
		char *Table;            // The table or view to which the trigger applies
		TK OP;					// One of TK_DELETE, TK_UPDATE, TK_INSERT
		TRIGGER TRtm;           // One of TRIGGER_BEFORE, TRIGGER_AFTER
		Expr *When;				// The WHEN clause of the expression (may be NULL)
		IdList *Columns;		// If this is an UPDATE OF <column-list> trigger, the <column-list> is stored here
		Schema *Schema;			// Schema containing the trigger
		Core::Schema *TabSchema; // Schema containing the table
		TriggerStep *StepList;	// Link list of trigger program steps
		Trigger *Next;			// Next trigger associated with the table

		__device__ static void DeleteTriggerStep(Context *ctx, TriggerStep *triggerStep);
		__device__ static Trigger *List(Parse *parse, Core::Table *table);
		__device__ static void BeginTrigger(Parse *parse,Token *name1, Token *name2, int trTm, int op, IdList *columns, SrcList *tableName, Expr *when, bool isTemp, int noErr);
		__device__ static void FinishTrigger(Parse *parse, TriggerStep *stepList, Token *all);
		__device__ static TriggerStep *SelectStep(Context *ctx, Select *select);
		__device__ static TriggerStep *InsertStep(Context *ctx, Token *tableName, IdList *column, ExprList *list, Select *select, OE orconf);
		__device__ static TriggerStep *UpdateStep(Context *ctx, Token *tableName, ExprList *list, Expr *where, OE orconf);
		__device__ static TriggerStep *DeleteStep(Context *ctx, Token *tableName, Expr *where_);
		__device__ static void DeleteTrigger(Context *ctx, Trigger *trigger);
		__device__ static void DropTrigger(Parse *parse, SrcList *name, int noErr);
		__device__ static void DropTriggerPtr(Parse *parse, Trigger *trigger);
		__device__ static void UnlinkAndDeleteTrigger(Context *ctx, int db, const char *name);
		__device__ static Trigger *TriggersExist(Parse *parse, Core::Table *table, int op, ExprList *changes, TRIGGER *maskOut);
		__device__ void CodeRowTriggerDirect(Parse *parse, Core::Table *table, int reg, OE orconf, int ignoreJump);
		__device__ void CodeRowTrigger(Parse *parse, TK op, ExprList *changes, TRIGGER trtm, ::Table *table, int reg, OE orconf, int ignoreJump);
		__device__ uint32 Colmask(Parse *parse, ExprList *changes, bool isNew, TRIGGER trtm, ::Table *table, OE orconf);
	};

	struct TriggerStep
	{
		TK OP;					// One of TK_DELETE, TK_UPDATE, TK_INSERT, TK_SELECT
		OE Orconf;				// OE_Rollback etc.
		Trigger *Trig;			// The trigger that this step is a part of
		Select *Select;			// SELECT statment or RHS of INSERT INTO .. SELECT ...
		Token Target;			// Target table for DELETE, UPDATE, INSERT
		Expr *Where;			// The WHERE clause for DELETE or UPDATE steps
		ExprList *ExprList;		// SET clause for UPDATE.  VALUES clause for INSERT
		IdList *IdList;			// Column names for INSERT
		TriggerStep *Next;		// Next in the link-list
		TriggerStep *Last;		// Last element in link-list. Valid for 1st elem only
	};

	struct TriggerPrg
	{
		Trigger *Trigger;		// Trigger this program was coded from
		TriggerPrg *Next;		// Next entry in Parse.pTriggerPrg list
		Vdbe::SubProgram *Program;	// Program implementing pTrigger/orconf
		OE Orconf;             // Default ON CONFLICT policy
		uint32 Colmasks[2];     // Masks of old.*, new.* columns accessed
	};

#pragma endregion

#pragma region Attach

	struct DbFixer
	{
		Parse *Parse;      // The parsing context.  Error messages written here
		Schema *Schema;    // Fix items to this schema
		const char *DB;    // Make sure all objects are contained in this database
		const char *Type;  // Type of the container - used for error messages
		const Token *Name; // Name of the container - used for error messages

		__device__ bool FixInit(Core::Parse *parse, int db, const char *typeName, const Token *name);
		__device__ bool FixSrcList(SrcList *list);
#if !defined(OMIT_VIEW) || !defined(OMIT_TRIGGER)
		__device__ bool FixSelect(Select *select);
		__device__ bool FixExpr(Expr *expr);
		__device__ bool FixExprList(ExprList *list);
#endif

#ifndef OMIT_TRIGGER
		__device__ bool FixTriggerStep(TriggerStep *step);
#endif
	};

#pragma endregion

#pragma region HasCodec
#ifdef HAS_CODEC
	struct Codec
	{
		__device__ static RC Attach(Context *ctx, int dbsLength, const void *key, int keyLength);
		__device__ static void GetKey(Context *ctx, int, void **key, int *keyLength);
		// pragma
		__device__ static int Key(Context *ctx, const void *key, int keyLength);
		__device__ static int Rekey(Context *ctx, const void *key, int keyLenegth);
		__device__ static void ActivateSee(const char *passPhrase);
#ifdef ENABLE_CEROD
		__device__ static void ActivateCerod(const char *passPhrase);
#endif
	};
#endif
#pragma endregion

#pragma region Command
	namespace Command {

		struct Alter
		{
			__device__ static void Functions();
			__device__ static void RenameTable(Parse *parse, SrcList *src, Token *name);
			__device__ static void MinimumFileFormat(Parse *parse, int db, int minFormat);
			__device__ static void FinishAddColumn(Parse *parse, Token *colDef);
			__device__ static void BeginAddColumn(Parse *parse, SrcList *src);
		};

		struct Analyze
		{
			__device__ static void Analyze_(Parse *parse, Token *name1, Token *name2);
			__device__ static void DeleteIndexSamples(Context *ctx, Index *idx);
			__device__ static RC AnalysisLoad(Context *ctx, int db);
		};

		struct Attach
		{
			__device__ static void Detach(Parse *parse, Expr *dbName);
			__device__ static void Attach_(Parse *parse, Expr *p, Expr *dbName, Expr *key);
		};

		struct Date_
		{
			__device__ static void RegisterDateTimeFunctions();
		};

		struct Delete
		{
			__device__ static Table *SrcListLookup(Parse *parse, SrcList *src);
			__device__ static bool IsReadOnly(Parse *parse, Table *table, bool viewOk);
#if !defined(OMIT_VIEW) && !defined(OMIT_TRIGGER)
			__device__ static void MaterializeView(Parse *parse, Table *view, Expr *where_, int curId);
#endif
#if 1 || defined(ENABLE_UPDATE_DELETE_LIMIT) && !defined(OMIT_SUBQUERY)
			__device__ static Expr *LimitWhere(Parse *parse, SrcList *src, Expr *where_, ExprList *orderBy, Expr *limit, Expr *offset, char *stmtType);
#endif
			__device__ static void DeleteFrom(Parse *parse, SrcList *tabList, Expr *where_);
			__device__ static void GenerateRowDelete(Parse *parse, Table *table, int curId, int rowid, int count, Trigger *trigger, OE onconf);
			__device__ static void GenerateRowIndexDelete(Parse *parse, Table *table, int curId, int *regIdxs);
			__device__ static int GenerateIndexKey(Parse *parse, Index *idx, int curId, int regOut, bool doMakeRec);
		};

		struct Func
		{
			__device__ static CollSeq *GetFuncCollSeq(FuncContext *fctx);
			__device__ static void SkipAccumulatorLoad(FuncContext *fctx);
			__device__ static void RegisterBuiltinFunctions(Context *ctx);
			__device__ static void RegisterLikeFunctions(Context *ctx, bool caseSensitive);
			__device__ static bool IsLikeFunction(Context *ctx, Expr *expr, bool *isNocase, char *wc);
			__device__ static void RegisterGlobalFunctions();
		};

		struct Insert
		{
			__device__ static void OpenTable(Parse *p, int cur, int db, Table *table, OP opcode);
			__device__ static const char *IndexAffinityStr(Vdbe *v, Index *index);
			__device__ static void TableAffinityStr(Vdbe *v, Table *table);
			__device__ static void AutoincrementBegin(Parse *parse);
			__device__ static void AutoincrementEnd(Parse *parse);
			__device__ static RC CodeCoroutine(Parse *parse, Select *select, SelectDest *dest);
			__device__ static void Insert_(Parse *parse, SrcList *tabList, ExprList *list, Select *select, IdList *column, OE onError);
			__device__ static void GenerateConstraintChecks(Parse *parse, Table *table, int baseCur, int regRowid, int *regIdxs, int rowidChng, int isUpdate, OE overrideError, int ignoreDest, bool *mayReplaceOut);
			__device__ static void CompleteInsertion(Parse *parse, Table *table, int baseCur, int regRowid, int *regIdxs, bool isUpdate, bool appendBias, bool useSeekResult);
			__device__ static int OpenTableAndIndices(Parse *parse, Table *table, int baseCur, OP op);
		};

		struct Pragma
		{
			__device__ static const char *JournalModename(IPager::JOURNALMODE mode);
			__device__ static void Pragma_(Parse *parse, Token *id1, Token *id2, Token *value, bool minusFlag);
		};

		//struct Select
		//{

		//};

		struct Update
		{
			__device__ static void ColumnDefault(Vdbe *v, Table *table, int i, int regId);
			__device__ static void Update_(Parse *parse, SrcList *tabList, ExprList *changes, Expr *where_, OE onError);
		};

		struct Vacuum
		{
			__device__ static void Vacuum_(Parse *parse);
			__device__ static RC RunVacuum(char **errMsg, Context *ctx);
		};

	}
#pragma endregion

#pragma region Main

	class Main
	{
	public:

#pragma region From: Util_c

		__device__ inline static RC ApiExit(Context *ctx, RC rc)
		{
			// If the ctx handle is not NULL, then we must hold the connection handle mutex here. Otherwise the read (and possible write) of db->mallocFailed is unsafe, as is the call to sqlite3Error().
			_assert(!ctx || MutexEx::Held(ctx->Mutex));
			if (ctx && (ctx->MallocFailed || rc == RC_IOERR_NOMEM))
			{
				Error(ctx, RC_NOMEM, nullptr);
				ctx->MallocFailed = false;
				rc = RC_NOMEM;
			}
			return (RC)(rc & (ctx ? ctx->ErrMask : 0xff));
		}

		__device__ static void Error(Context *ctx, RC errCode, const char *fmt, va_list *args);
#if __CUDACC__
		__device__ inline static void Error(Context *ctx, RC errCode, const char *fmt) { va_list args; va_start(args, nullptr); Error(ctx, errCode, fmt, &args); va_end(args); }
		template <typename T1> __device__ inline static void Error(Context *ctx, RC errCode, const char *fmt, T1 arg1) { va_list args; va_start(args, arg1); Error(ctx, errCode, fmt, &args); va_end(args); }
		template <typename T1, typename T2> __device__ inline static void Error(Context *ctx, RC errCode, const char *fmt, T1 arg1, T2 arg2) { va_list args; va_start(args, arg1, arg2); Error(ctx, errCode, fmt, &args); va_end(args); }
		template <typename T1, typename T2, typename T3> __device__ inline static void Error(Context *ctx, RC errCode, const char *fmt, T1 arg1, T2 arg2, T3 arg3) { va_list args; va_start(args, arg1, arg2, arg3); Error(ctx, errCode, fmt, &args); va_end(args); }
		template <typename T1, typename T2, typename T3, typename T4> __device__ inline static void Error(Context *ctx, RC errCode, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list args; va_start(args, arg1, arg2, arg3, arg4); Error(ctx, errCode, fmt, &args); va_end(args); }
		template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ inline static void Error(Context *ctx, RC errCode, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5); Error(ctx, errCode, fmt, &args); va_end(args); }
		template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6> __device__ inline static void Error(Context *ctx, RC errCode, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6); Error(ctx, errCode, fmt, &args); va_end(args); }
		template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7> __device__ inline static void Error(Context *ctx, RC errCode, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7); Error(ctx, errCode, fmt, &args); va_end(args); }
		template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8> __device__ inline static void Error(Context *ctx, RC errCode, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8); Error(ctx, errCode, fmt, &args); va_end(args); }
		template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9> __device__ inline static void Error(Context *ctx, RC errCode, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); Error(ctx, errCode, fmt, &args); va_end(args); }
		template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename TA> __device__ inline static void Error(Context *ctx, RC errCode, const char *fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, TA argA) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, argA); Error(ctx, errCode, fmt, &args); va_end(args); }
#else
		__device__ inline void static Error(Context *ctx, RC errCode, const char *fmt, ...) { va_list args; va_start(args, fmt); Error(ctx, errCode, fmt, &args); va_end(args); }
#endif
		__device__ static bool SafetyCheckOk(Context *ctx);
		__device__ static bool SafetyCheckSickOrOk(Context *ctx);

#pragma endregion

#pragma region From: Legacy_c

		__device__ static RC Exec(Context *ctx, const char *sql, bool (*callback)(void *, int, char **, char **), void *arg, char **errmsg);

#pragma endregion

#pragma region From: LoadExt_c

		__device__ static RC LoadExtension_(Context *ctx, const char *fileName, const char *procName, char **errMsgOut);
		__device__ static RC LoadExtension(Context *ctx, const char *fileName, const char *procName, char **errMsgOut);
		__device__ static void CloseExtensions(Context *ctx);
		__device__ static RC EnableLoadExtension(Context *ctx, bool onoff);
		__device__ static RC AutoExtension(void (*init)());
		__device__ static void ResetAutoExtension();
		__device__ static void AutoLoadExtensions(Context *ctx);

#pragma endregion

#pragma region Initialize/Shutdown/Config

		struct GlobalStatics
		{
			bool UseCis;						// Use covering indices for full-scans
			//sqlite3_pcache_methods2 pcache2;	// Low-level page-cache interface
			void *Page;							// Page cache memory
			int PageSize;						// Size of each page in pPage[]
			int Pages;							// Number of pages in pPage[]
			int MaxParserStack;					// maximum depth of the parser stack
			// The above might be initialized to non-zero.  The following need to always initially be zero, however.
			bool IsPCacheInit;					// True after malloc is initialized
		};

		__device__ static RC Initialize();
		__device__ static RC Shutdown();

		enum CONFIG
		{
			CONFIG_PAGECACHE = 7,				// void*, int sz, int N
			CONFIG_PCACHE = 14,					// no-op
			CONFIG_GETPCACHE = 15,				// no-op
			CONFIG_PCACHE2 = 18,				// sqlite3_pcache_methods2*
			CONFIG_GETPCACHE2 = 19,				// sqlite3_pcache_methods2*
			CONFIG_COVERING_INDEX_SCAN = 20,	// int
		};
		__device__ static RC Config(CONFIG op, va_list *args);
#if __CUDACC__
		__device__ inline static RC Config(CONFIG op) { va_list args; va_start(args, nullptr); RC r = Config(op, &args); va_end(args); return r; }
		template <typename T1> __device__ inline static RC Config(CONFIG op, T1 arg1) { va_list args; va_start(args, arg1); RC r = Config(op, &args); va_end(args); return r; }
		template <typename T1, typename T2> __device__ inline static RC Config(CONFIG op, T1 arg1, T2 arg2) { va_list args; va_start(args, arg1, arg2); RC r = Config(op, &args); va_end(args); return r; }
		template <typename T1, typename T2, typename T3> __device__ inline static RC Config(CONFIG op, T1 arg1, T2 arg2, T3 arg3) { va_list args; va_start(args, arg1, arg2, arg3); RC r = Config(op, &args); va_end(args); return r; }
		template <typename T1, typename T2, typename T3, typename T4> __device__ inline static RC Config(CONFIG op, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list args; va_start(args, arg1, arg2, arg3, arg4); RC r = Config(op, &args); va_end(args); return r; }
		template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ inline static RC Config(CONFIG op, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5); RC r = Config(op, &args); va_end(args); return r; }
#else
		__device__ inline static RC Config(CONFIG op, ...) { va_list args; va_start(args, op); RC r = Config(op, &args); va_end(args); return r; }
#endif

		enum CTXCONFIG
		{
			CTXCONFIG_LOOKASIDE = 1001,  // void* int int
			CTXCONFIG_ENABLE_FKEY = 1002,  // int int*
			CTXCONFIG_ENABLE_TRIGGER = 1003,  // int int*
		};
		__device__ static RC CtxConfig(Context *ctx, CTXCONFIG op, va_list *args);
#if __CUDACC__
		__device__ inline static RC CtxConfig(Context *ctx, CTXCONFIG op) { va_list args; va_start(args, nullptr); RC r = CtxConfig(ctx, op, &args); va_end(args); return r; }
		template <typename T1> __device__ inline static RC CtxConfig(Context *ctx, CTXCONFIG op, T1 arg1) { va_list args; va_start(args, arg1); RC r = CtxConfig(ctx, op, &args); va_end(args); return r; }
		template <typename T1, typename T2> __device__ inline static RC CtxConfig(Context *ctx, CTXCONFIG op, T1 arg1, T2 arg2) { va_list args; va_start(args, arg1, arg2); RC r = CtxConfig(ctx, op, &args); va_end(args); return r; }
		template <typename T1, typename T2, typename T3> __device__ inline static RC CtxConfig(Context *ctx, CTXCONFIG op, T1 arg1, T2 arg2, T3 arg3) { va_list args; va_start(args, arg1, arg2, arg3); RC r = CtxConfig(ctx, op, &args); va_end(args); return r; }
		template <typename T1, typename T2, typename T3, typename T4> __device__ inline static RC CtxConfig(Context *ctx, CTXCONFIG op, T1 arg1, T2 arg2, T3 arg3, T4 arg4) { va_list args; va_start(args, arg1, arg2, arg3, arg4); RC r = CtxConfig(ctx, op, &args); va_end(args); return r; }
		template <typename T1, typename T2, typename T3, typename T4, typename T5> __device__ inline static RC CtxConfig(Context *ctx, CTXCONFIG op, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) { va_list args; va_start(args, arg1, arg2, arg3, arg4, arg5); RC r = CtxConfig(ctx, op, &args); va_end(args); return r; }
#else
		__device__ inline static RC CtxConfig(Context *ctx, CTXCONFIG op, ...) { va_list args; va_start(args, op); RC r = CtxConfig(ctx, op, &args); va_end(args); return r; }
#endif

#pragma endregion

		__device__ static RC CtxReleaseMemory(Context *ctx);
		__device__ static void CloseSavepoints(Context *ctx);
		__device__ static RC Close(Context *ctx, bool forceZombie);
		__device__ static RC Close(Context *ctx);
		__device__ static RC Close_v2(Context *ctx);
		__device__ static void LeaveMutexAndCloseZombie(Context *ctx);
		__device__ static void RollbackAll(Context *ctx, RC tripCode);
		__device__ static const char *ErrStr(RC rc);
		__device__ static int DefaultBusyCallback(void *ptr, int count);
		__device__ static int InvokeBusyHandler(Context::BusyHandlerType *p);
		__device__ static RC BusyHandler(Context *ctx, int (*busy)(void *, int), void *arg);
#ifndef OMIT_PROGRESS_CALLBACK
		__device__ static void ProgressHandler(Context *ctx,  int ops, int (*progress)(void *), void *arg);
#endif
		__device__ static RC BusyTmeout(Context *ctx, int ms);
		__device__ static void Interrupt(Context *ctx);
		__device__ static RC CreateFunc(Context *ctx, const char *funcName, int args, TEXTENCODE encode, void *userData, void (*func)(FuncContext*,int,Mem**), void (*step)(FuncContext*,int,Mem**), void (*final_)(FuncContext*), FuncDestructor *destructor);
		__device__ static RC CreateFunction(Context *ctx, const char *funcName, int args, TEXTENCODE encode, void *p, void (*func)(FuncContext*,int,Mem**), void (*step)(FuncContext*,int,Mem**), void (*final_)(FuncContext*));
		__device__ static RC CreateFunction_v2(Context *ctx, const char *funcName, int args, TEXTENCODE encode, void *p, void (*func)(FuncContext*,int,Mem**), void (*step)(FuncContext*,int,Mem**), void (*final_)(FuncContext*), void (*destroy)(void*));
#ifndef OMIT_UTF16
		__device__ static RC CreateFunction16(Context *ctx, const void *funcName, int args, TEXTENCODE encode, void *p, void (*func)(FuncContext*,int,Mem**), void (*step)(FuncContext*,int,Mem**), void (*final_)(FuncContext*));
#endif
		__device__ static RC OverloadFunction( Context *ctx, const char *funcName, int args);
#ifndef OMIT_TRACE
		__device__ static void *Trace(Context *ctx, void (*trace)(void*,const char*), void *arg);
		__device__ static void *Profile(Context *ctx, void (*profile)(void*,const char*,uint64), void *arg);
#endif
		__device__ static void *CommitHook(Context *ctx, RC (*callback)(void*), void *arg);
		__device__ static void *UpdateHook(Context *ctx, void (*callback)(void*,int,char const*,char const*,int64), void *arg);
		__device__ static void *RollbackHook(Context *ctx, void (*callback)(void*), void *arg);
#ifndef OMIT_WAL
		__device__ static RC WalDefaultHook(void *clientData, Context *ctx, const char *dbName, int frames);
#endif
		__device__ static RC WalAutocheckpoint(Context *ctx, int frames);
		__device__ static void *WalHook(Context *ctx, int (*callback)(void*,Context*,const char*,int), void *arg);
		__device__ static RC WalCheckpoint(Context *ctx, const char *dbName);
		__device__ static RC WalCheckpoint_v2(Context *ctx, const char *dbName, IPager::CHECKPOINT mode, int *logsOut, int *ckptsOut);
#ifndef OMIT_WAL
		__device__ static RC Checkpoint(Context *ctx, int db, IPager::CHECKPOINT mode, int *logsOut, int *ckptsOut);
#endif
		// BContext::TempInMemory -> __device__ static bool TempInMemory(Context *ctx);
		__device__ static const char *ErrMsg(Context *ctx);
		__device__ static const void *ErrMsg16(Context *ctx);
		__device__ static RC ErrCode(Context *ctx);
		__device__ static RC ExtendedErrCode(Context *ctx);
		__device__ static int Limit(Context *ctx, LIMIT limit, int newLimit);
		__device__ static RC Open(const char *fileName, Context **ctxOut);
		__device__ static RC Open_v2(const char *fileName, Context **ctxOut, VSystem::OPEN flags, const char *vfsName);
#ifndef OMIT_UTF16
		__device__ static RC Open16(const void *fileName,  Context **ctxOut);
#endif
		__device__ static RC CreateCollation(Context *ctx, const char *name, TEXTENCODE encode, void *ctx2, int (*compare)(void*,int,const void*,int,const void*));
		__device__ static RC CreateCollation_v2(Context *ctx, const char *name, TEXTENCODE encode, void *ctx2, int (*compare)(void*,int,const void*,int,const void*), void (*del)(void*));
#ifndef OMIT_UTF16
		__device__ static RC CreateCollation16(Context *ctx, const void *name, TEXTENCODE encode,  void *ctx2, int (*compare)(void*,int,const void*,int,const void*));
#endif
		__device__ static RC CollationNeeded(Context *ctx, void *collNeededArg, void (*collNeeded)(void*,Context*,TEXTENCODE,const char*));
#ifndef OMIT_UTF16
		__device__ static RC CollationNeeded16(Context *ctx, void *collNeededArg, void (*collNeeded16)(void*,Context*,TEXTENCODE,const void*));
#endif
#ifdef ENABLE_COLUMN_METADATA
		__device__ static RC TableColumnMetadata(Context *ctx, const char *dbName, const char *tableName, const char *columnName, char const **dataTypeOut, char const **collSeqNameOut, bool *notNullOut, bool *primaryKeyOut, bool *autoincOut);
#endif
		__device__ static int Sleep(int ms);
		__device__ static RC ExtendedResultCodes(Context *ctx, bool onoff);
		__device__ static RC FileControl(Context *ctx, const char *dbName, VFile::FCNTL op, void *arg);
		enum TESTCTRL
		{
			TESTCTRL_FIRST                   = 5,
			TESTCTRL_PRNG_SAVE               = 5,
			TESTCTRL_PRNG_RESTORE            = 6,
			TESTCTRL_PRNG_RESET              = 7,
			TESTCTRL_BITVEC_TEST             = 8,
			TESTCTRL_FAULT_INSTALL           = 9,
			TESTCTRL_BENIGN_MALLOC_HOOKS     =10,
			TESTCTRL_PENDING_BYTE            =11,
			TESTCTRL_ASSERT                  =12,
			TESTCTRL_ALWAYS                  =13,
			TESTCTRL_RESERVE                 =14,
			TESTCTRL_OPTIMIZATIONS           =15,
			TESTCTRL_ISKEYWORD               =16,
			TESTCTRL_SCRATCHMALLOC           =17,
			TESTCTRL_LOCALTIME_FAULT         =18,
			TESTCTRL_EXPLAIN_STMT            =19,
			TESTCTRL_LAST                    =19,
		};
		__device__ RC Main::TestControl(TESTCTRL op, va_list *args);
		__device__ static Btree *DbNameToBtree(Context *ctx, const char *dbName);
		__device__ static const char *CtxFilename(Context *ctx, const char *dbName);
		__device__ static int CtxReadonly(Context *ctx, const char *dbName);

		// inlined
		__device__ inline static int64 CtxLastInsertRowid(Context *ctx) { return ctx->LastRowID; }
		__device__ inline static int CtxChanges(Context *ctx) { return ctx->Changes; }
		__device__ inline static int CtxTotalChanges(Context *ctx) { return ctx->TotalChanges; }
	};

	__device__ extern _WSD Main::GlobalStatics g_globalStatics;
	__device__ extern _WSD FuncDefHash g_globalFunctions;
#define Main_GlobalStatics _GLOBAL(Main::GlobalStatics, g_globalStatics)
#define Main_GlobalFunctions _GLOBAL(FuncDefHash, g_globalFunctions)

#pragma endregion
}
using namespace Core::Command;
