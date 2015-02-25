using System;
#region Limit Types
#if MAX_ATTACHED
    using yDbMask = System.UInt64;
#else
using yDbMask = System.UInt32;
#endif
#if MAX_VARIABLE_NUMBER
    using yVars = System.Int16;
#else
using yVars = System.Int32;
#endif
#if _64BITSTATS
	using tRowcnt = System.UInt64; // 64-bit only if requested at compile-time
#else
using tRowcnt = System.UInt32; // 32-bit is the default
#endif
using Bitmask = System.UInt64;
using Core;
using System.Text;
#endregion
namespace Core
{
    #region Limits
    public partial class L
    {
        public const int MAX_ATTACHED = 10;
        public const int N_COLCACHE = 10;
        //#define BMS ((int)(sizeof(Bitmask)*8))
#if !MAX_EXPR_DEPTH
        public const int MAX_EXPR_DEPTH = 1000;
#endif
#if OMIT_TEMPDB
        public const bool OMIT_TEMPDB = true;
#else
        public const bool OMIT_TEMPDB = false;
#endif
    }
    #endregion

    #region Func

    enum FUNC : byte
    {
        LIKE = 0x01,			// Candidate for the LIKE optimization
        CASE = 0x02,			// Case-sensitive LIKE-type function
        EPHEM = 0x04,			// Ephemeral.  Delete with VDBE
        NEEDCOLL = 0x08,		// sqlite3GetFuncCollSeq() might be called
        COUNT = 0x10,			// Built-in count(*) aggregate
        COALESCE = 0x20,		// Built-in coalesce() or ifnull() function
        LENGTH = 0x40,			// Built-in length() function
        TYPEOF = 0x80,			// Built-in typeof() function
    }

    public class FuncDestructor
    {
        public int Refs;
        public Action<object> Destroy;
        public object UserData;
    }

    public class FuncDef
    {
        public short Args;					// Number of arguments.  -1 means unlimited
        public TEXTENCODE PrefEncode;		// Preferred text encoding (SQLITE_UTF8, 16LE, 16BE)
        public FUNC Flags;					// Some combination of SQLITE_FUNC_*
        public object UserData;				// User data parameter
        public FuncDef Next;				// Next function with same name */
        public Action<FuncContext, int, Mem[]> Func; // Regular function
        public Action<FuncContext, int, Mem[]> Step; // Aggregate step
        public Action<FuncContext> Finalize; // Aggregate finalizer
        public string Name;					// SQL name of the function.
        public FuncDef Hash;				// Next with a different name but the same hash
        public FuncDestructor Destructor; // Reference counted destructor function

        internal FuncDef _memcpy()
        {
            FuncDef c = new FuncDef();
            c.Args = Args;
            c.PrefEncode = PrefEncode;
            c.Flags = Flags;
            c.UserData = UserData;
            c.Next = Next;
            c.Func = Func;
            c.Step = Step;
            c.Finalize = Finalize;
            c.Name = Name;
            c.Hash = Hash;
            c.Destructor = Destructor;
            return c;
        }
    }

    //   FUNCTION(zName, nArg, iArg, bNC, xFunc)
    //     Used to create a scalar function definition of a function zName implemented by C function xFunc that accepts nArg arguments. The
    //     value passed as iArg is cast to a (void*) and made available as the user-data (sqlite3_user_data()) for the function. If 
    //     argument bNC is true, then the SQLITE_FUNC_NEEDCOLL flag is set.
    //
    //   AGGREGATE(zName, nArg, iArg, bNC, xStep, xFinal)
    //     Used to create an aggregate function definition implemented by the C functions xStep and xFinal. The first four parameters
    //     are interpreted in the same way as the first 4 parameters to FUNCTION().
    //
    //   LIKEFUNC(zName, nArg, pArg, flags)
    //     Used to create a scalar function definition of a function zName that accepts nArg arguments and is implemented by a call to C 
    //     function likeFunc. Argument pArg is cast to a (void *) and made available as the function user-data (sqlite3_user_data()). The
    //     FuncDef.flags variable is set to the value passed as the flags parameter.
    //#define FUNCTION(zName, nArg, iArg, bNC, xFunc) {nArg, SQLITE_UTF8, (bNC*FUNC_NEEDCOLL), \
    //    INT_TO_PTR(iArg), 0, xFunc, 0, 0, #zName, 0, 0}
    //#define FUNCTION2(zName, nArg, iArg, bNC, xFunc, extraFlags) {nArg, SQLITE_UTF8, (bNC*FUNC_NEEDCOLL)|extraFlags, \
    //    INT_TO_PTR(iArg), 0, xFunc, 0, 0, #zName, 0, 0}
    //#define STR_FUNCTION(zName, nArg, pArg, bNC, xFunc) {nArg, SQLITE_UTF8, bNC*FUNC_NEEDCOLL, \
    //    pArg, 0, xFunc, 0, 0, #zName, 0, 0}
    //#define LIKEFUNC(zName, nArg, arg, flags) {nArg, TEXTENCODE_UTF8, flags, (void *)arg, 0, likeFunc, 0, 0, #zName, 0, 0}
    //#define AGGREGATE(zName, nArg, arg, nc, xStep, xFinal) {nArg, TEXTENCODE_UTF8, nc*SQLITE_FUNC_NEEDCOLL, \
    //    INT_TO_PTR(arg), 0, 0, xStep,xFinal,#zName,0,0}

    #endregion

    #region ITable

    public class TableModule
    {
        public ITableModule IModule;        // Callback pointers
        public string Name;                 // Name passed to create_module()
        public object Aux;                  // pAux passed to create_module()
        public Action<object> Destroy;      // Module destructor function
    }

    public class ITableModule
    {
        public int Version;
        public Func<Context, object, int, string[], VTable[], string[], RC> Create;
        public Func<Context, object, int, string[], VTable[], string[]> Connect;
        public Func<VTable, IIndexInfo, RC> BestIndex;
        public Func<VTable, RC> Disconnect;
        public Func<VTable, RC> Destroy;
        public Func<VTable, IVTableCursor[], RC> Open;
        public Func<IVTableCursor, RC> Close;
        public Func<IVTableCursor, int, string, int, Mem[], RC> Filter;
        public Func<IVTableCursor, RC> Next;
        public Func<IVTableCursor, RC> Eof;
        public Func<IVTableCursor, FuncContext, int, RC> Column;
        public Func<IVTableCursor, long[], RC> Rowid;
        public Func<VTable, int, Mem[], long[], RC> Update;
        public Func<IVTable, RC> Begin;
        public Func<IVTable, RC> Sync;
        public Func<IVTable, RC> Commit;
        public Func<IVTable, RC> Rollback;
        public Func<VTable, int, string, Action<FuncContext, int, Mem[]>, object[], RC> FindFunction;
        public Func<VTable, string, RC> Rename;
        public Func<VTable, int, RC> Savepoint;
        public Func<VTable, int, RC> Release;
        public Func<VTable, int, RC> RollbackTo;
    }

    public enum INDEX_CONSTRAINT : byte
    {
        EQ = 2,
        GT = 4,
        LE = 8,
        LT = 16,
        GE = 32,
        MATCH = 64,
    }

    public class IIndexInfo
    {
        public struct Constraint
        {
            public int Column;         // Column on left-hand side of constraint
            public INDEX_CONSTRAINT OP;// Constraint operator
            public bool Usable;		// True if this constraint is usable
            public int TermOffset;     // Used internally - xBestIndex should ignore
        }
        public struct Orderby
        {
            public int Column;			// Column number
            public bool Desc;			// True for DESC.  False for ASC.
        }
        public struct ConstraintUsage
        {
            public int ArgvIndex;		// if >0, constraint is part of argv to xFilter
            public byte Omit; // Do not code a test for this constraint
        }
        // INPUTS
        public array_t<Constraint> Constraints; // Table of WHERE clause constraints
        public array_t<Orderby> OrderBys; // The ORDER BY clause
        // OUTPUTS
        public array_t<ConstraintUsage> ConstraintUsages;
        public int IdxNum;              // Number used to identify the index
        public string IdxStr;           // String, possibly obtained from sqlite3_malloc
        public bool NeedToFreeIdxStr;    // Free idxStr using sqlite3_free() if true
        public bool OrderByConsumed;    // True if output is already ordered
        public double EstimatedCost;    // Estimated cost of using this index
    }

    public class IVTable
    {
        public ITableModule IModule;	// The module for this virtual table
        public string ErrMsg;			// Error message from sqlite3_mprintf()
        // Virtual table implementations will typically add additional fields
    }

    public class IVTableCursor
    {
        public IVTable IVTable;		    // Virtual table of this cursor
        // Virtual table implementations will typically add additional fields
    }

    #endregion

    #region Types

    public enum OE : byte
    {
        None = 0,   // There is no constraint to check
        Rollback = 1,   // Fail the operation and rollback the transaction
        Abort = 2,   // Back out changes but do no rollback transaction
        Fail = 3,   // Stop the operation but leave all prior changes
        Ignore = 4,   // Ignore the error. Do not do the INSERT or UPDATE
        Replace = 5,   // Delete existing record, then do INSERT or UPDATE
        Restrict = 6,   // OE_Abort for IMMEDIATE, OE_Rollback for DEFERRED
        SetNull = 7,   // Set the foreign key value to NULL
        SetDflt = 8,   // Set the foreign key value to its default
        Cascade = 9,   // Cascade the changes
        Default = 99,  // Do whatever the default action is
    }

    public enum CONFLICT : byte
    {
        ROLLBACK = 1,
        IGNORE = 2,
        FAIL = 3,
        ABORT = 4,
        REPLACE = 5
    }

    public enum TYPE : byte
    {
        INTEGER = 1,
        FLOAT = 2,
        BLOB = 4,
        NULL = 5,
        TEXT = 3,
    }

    public class Index
    {
        public string Name;				// Name of this index
        public array_t2<ushort, int> Columns;   // Which columns are used by this index.  1st is 0 // Number of columns in table used by this index
        public tRowcnt[] RowEsts;		// From ANALYZE: Est. rows selected by each column
        public Table Table;				// The SQL table being indexed
        public string ColAff;			// String defining the affinity of each column
        public Index Next;				// The next index associated with the same table
        public Schema Schema;			// Schema containing this index
        public SO[] SortOrders;			// for each column: True==DESC, False==ASC
        public string[] CollNames;		// Array of collation sequence names for index
        public int Id;					// DB Page containing root of this index
        public OE OnError;			    // OE_Abort, OE_Ignore, OE_Replace, or OE_None
        public byte AutoIndex;		    // 1==UNIQUE, 2==PRIMARY KEY, 0==CREATE INDEX
        public bool Unordered;		    // Use this index for == or IN queries only
#if ENABLE_STAT3
        public tRowcnt AvgEq;					// Average nEq value for key values not in aSample
        public array_t<IndexSample> Samples;   // Samples of the left-most key
#endif
    }

    public class IndexSample
    {
        public class _u
        {
            public string Z;			// Value if eType is SQLITE_TEXT or SQLITE_BLOB */
            public double R;			// Value if eType is SQLITE_FLOAT */
            public long I;			    // Value if eType is SQLITE_INTEGER */
        }
        public _u u;
        public TYPE Type;				// SQLITE_NULL, SQLITE_INTEGER ... etc.
        public int Bytes;				// Size in byte of text or blob.
        public tRowcnt Eqs;			    // Est. number of rows where the key equals this sample
        public tRowcnt Lts;			    // Est. number of rows where key is less than this sample
        public tRowcnt DLts;			// Est. number of distinct keys less than this sample
    }

    public class Token
    {
        public string data;             // Text of the token.  Not NULL-terminated!
        public uint length;             // Number of characters in this token
    }

    public class AggInfo
    {
        public class AggInfoColumn
        {
            public Table Table;         // Source table
            public int TableID;         // Cursor number of the source table
            public int Column;			// Column number within the source table
            public int SorterColumn;	// Column number in the sorting index
            public int Mem;				// Memory location that acts as accumulator
            public Expr Expr;			// The original expression
        }
        public class AggInfoFunc
        {
            public Expr Expr;           // Expression encoding the function
            public FuncDef Func;        // The aggregate function implementation
            public int Mem;             // Memory location that acts as accumulator
            public int Distinct;        // Ephemeral table used to enforce DISTINCT
        }
        public byte DirectMode;			// Direct rendering mode means take data directly from source tables rather than from accumulators
        public byte UseSortingIdx;		// In direct mode, reference the sorting index rather than the source table
        public int SortingIdx;			// Cursor number of the sorting index
        public int SortingIdxPTab;		// Cursor number of pseudo-table
        public int SortingColumns;		// Number of columns in the sorting index
        public ExprList GroupBy;		// The group by clause
        public array_t<AggInfoColumn> Columns; // For each column used in source tables
        public int Accumulators;		// Number of columns that show through to the output. Additional columns are used only as parameters to aggregate functions
        public array_t<AggInfoFunc> Funcs; // For each aggregate function
    }

    public enum AFF : byte
    {
        TEXT = 'a',
        NONE = 'b',
        NUMERIC = 'c',
        INTEGER = 'd',
        REAL = 'e',
        MASK = 0x67, // The SQLITE_AFF_MASK values masks off the significant bits of an affinity value. 
        // Additional bit values that can be ORed with an affinity without changing the affinity.
        BIT_JUMPIFNULL = 0x08, // jumps if either operand is NULL
        BIT_STOREP2 = 0x10,	// Store result in reg[P2] rather than jump
        BIT_NULLEQ = 0x80,  // NULL=NULL
    }

    public partial class E
    {
        public static bool IsNumericAffinity(this AFF X) { return (X >= AFF.NUMERIC); }
    }

    #endregion

    #region Select

    public class IdList
    {
        public class IdListItem
        {
            public string Name;     // Name of the identifier
            public int Idx;         // Index in some Table.aCol[] of a column named zName
        }
        public array_t<IdListItem> Ids;

        internal IdList memcpy()
        {
            if (this == null)
                return null;
            IdList cp = (IdList)MemberwiseClone();
            Ids.data.CopyTo(cp.Ids.data, 0);
            cp.Ids.length = Ids.length;
            return cp;
        }
    }

    public enum JT : byte
    {
        INNER = 0x0001,     // Any kind of inner or cross join
        CROSS = 0x0002,     // Explicit use of the CROSS keyword
        NATURAL = 0x0004,   // True for a "natural" join
        LEFT = 0x0008,      // Left outer join
        RIGHT = 0x0010,     // Right outer join
        OUTER = 0x0020,     // The "OUTER" keyword is present
        ERROR = 0x0040,     // unknown or unsupported join type
    }

    public class SrcList
    {
        public class SrcListItem
        {
            public Schema Schema;		// Schema to which this item is fixed
            public string Database;		// Name of database holding this table
            public string Name;			// Name of the table
            public string Alias;		// The "B" part of a "A AS B" phrase.  zName is the "A"
            public Table Table;		    // An SQL table corresponding to zName
            public Select Select;		// A SELECT statement used in place of a table name
            public int AddrFillSub;	    // Address of subroutine to manifest a subquery
            public int RegReturn;		// Register holding return address of addrFillSub
            public JT Jointype;		    // Type of join between this able and the previous
            public bool NotIndexed;     // True if there is a NOT INDEXED clause
            public bool IsCorrelated;   // True if sub-query is correlated
            public bool ViaCoroutine;   // Implemented as a co-routine
#if !OMIT_EXPLAIN
            public byte SelectId;       // If pSelect!=0, the id of the sub-select in EQP
#endif
            public int Cursor;			// The VDBE cursor number used to access this table
            public Expr On;			    // The ON clause of a join
            public IdList Using;		// The USING clause of a join
            public Bitmask ColUsed;	    // Bit N (1<<N) set if column N of pTab is used
            public string IndexName;    // Identifier from "INDEXED BY <zIndex>" clause
            public Index Index;		    // Index structure corresponding to zIndex, if any
        }
        public short Srcs;				// Number of tables or subqueries in the FROM clause
        public short Allocs;			// Number of entries allocated in a[] below
        public SrcListItem[] Ids = new SrcListItem[1];		// One entry for each identifier on the list
    }

    public enum WHERE : uint
    {
        ORDERBY_NORMAL = 0x0000,	// No-op
        ORDERBY_MIN = 0x0001,		// ORDER BY processing for min() func
        ORDERBY_MAX = 0x0002,		// ORDER BY processing for max() func
        ONEPASS_DESIRED = 0x0004,   // Want to do one-pass UPDATE/DELETE
        DUPLICATES_OK = 0x0008,	    // Ok to return a row more than once
        OMIT_OPEN_CLOSE = 0x0010,   // Table cursors are already open
        FORCE_TABLE = 0x0020,		// Do not use an index-only search
        ONETABLE_ONLY = 0x0040,	    // Only code the 1st table in pTabList
        AND_ONLY = 0x0080,		    // Don't use indices for OR terms
    }

    public struct WherePlan
    {
        public WHERE WsFlags;       // WHERE_* flags that describe the strategy
        public ushort Eqs;          // Number of == constraints
        public ushort OBSats;       // Number of ORDER BY terms satisfied
        public double Rows;			// Estimated number of rows (for EQP)
        public class _u
        {
            public Index Index;     // Index when WHERE_INDEXED is true
            public WhereTerm Term;  // WHERE clause term for OR-search
            public IIndexInfo VTableIndex;	// Virtual table index to use
        }
        public _u u;

        internal void memset()
        {
            WsFlags = 0;
            Eqs = 0;
            OBSats = 0;
            Rows = 0;
            u.Index = null;
            u.Term = null;
            u.VTableIndex = null;
        }
    }

    public struct WhereLevel
    {
        public class InLoop
        {
            public int Cur;			    // The VDBE cursor used by this IN operator
            public int AddrInTop;		// Top of the IN loop
            public OP EndLoopOp;	    // IN Loop terminator. OP_Next or OP_Prev
        }
        public WherePlan Plan;		// query plan for this element of the FROM clause
        public int LeftJoin;		// Memory cell used to implement LEFT OUTER JOIN
        public int TabCur;			// The VDBE cursor used to access the table
        public int IdxCur;			// The VDBE cursor used to access pIdx
        public int AddrBrk;			// Jump here to break out of the loop
        public int AddrNxt;			// Jump here to start the next IN combination
        public int AddrCont;		// Jump here to continue with the next loop cycle
        public int AddrFirst;		// First instruction of interior of the loop
        public byte From;			// Which entry in the FROM clause
        public byte OP, P5;         // Opcode and P5 of the opcode that ends the loop
        public int P1, P2;			// Operands of the opcode used to ends the loop
        public class _u
        {
            public class _in
            {
                public InLoop[] InLoops; // Information about each nested IN operator
                public int InLoopsLength;
            }
            public _in in_;						// Used when plan.wsFlags&WHERE_IN_ABLE
            public Index Covidx;				// Possible covering index for WHERE_MULTI_OR
        }
        public _u u;							// Information that depends on plan.wsFlags
        public double OptCost;					// "Optimal" cost for this level
        // The following field is really not part of the current level.  But we need a place to cache virtual table index information for each
        // virtual table in the FROM clause and the WhereLevel structure is a convenient place since there is one WhereLevel for each FROM clause element.
        public IIndexInfo IndexInfo;			// Index info for n-th source table
    }

    public enum WHERE_DISTINCT : byte
    {
        NOOP = 0,		    // DISTINCT keyword not used
        UNIQUE = 1,		    // No duplicates
        ORDERED = 2,		// All duplicates are adjacent
        UNORDERED = 3,	    // Duplicates are scattered
    }

    public class WhereInfo
    {
        public Parse Parse;				    // Parsing and code generating context
        public SrcList TabList;			    // List of tables in the join
        public ushort OBSats;               // Number of ORDER BY terms satisfied by indices
        public WHERE WctrlFlags;            // Flags originally passed to sqlite3WhereBegin()
        public bool OkOnePass;              // Ok to use one-pass algorithm for UPDATE/DELETE
        public byte UntestedTerms;          // Not all WHERE terms resolved by outer loop
        public WHERE_DISTINCT eDistinct;    // One of the WHERE_DISTINCT_* values below
        public int TopId;					// The very beginning of the WHERE loop
        public int ContinueId;				// Jump here to continue with next record
        public int BreakId;					// Jump here to break out of the loop
        public int Levels;					// Number of nested loop
        public Where.WhereClause WC;			    // Decomposition of the WHERE clause
        public double SavedNQueryLoop;		// pParse->nQueryLoop outside the WHERE loop
        public double RowOuts;				// Estimated number of output rows
        public WhereLevel[] Data = new WhereLevel[1];	// Information about each nest loop in WHERE
    }

    public enum NC : byte
    {
        AllowAgg = 0x01,			// Aggregate functions are allowed here
        HasAgg = 0x02,			    // One or more aggregate functions seen
        IsCheck = 0x04,			    // True if resolving names in a CHECK constraint
        InAggFunc = 0x08,		    // True if analyzing arguments to an agg func
    }

    public class NameContext
    {
        public Parse Parse;				// The parser
        public SrcList SrcList;			// One or more tables used to resolve names
        public ExprList EList;			// Optional list of named expressions
        public AggInfo AggInfo;			// Information about aggregates at this level
        public NameContext Next;		// Next outer name context.  NULL for outermost
        public int Refs;				// Number of names resolved by this context
        public int Errs;				// Number of errors encountered while resolving names
        public NC NCFlags;				// Zero or more NC_* flags defined below
    }

    public enum SF : ushort
    {
        Distinct = 0x0001,		    // Output should be DISTINCT
        Resolved = 0x0002,		    // Identifiers have been resolved
        Aggregate = 0x0004,		    // Contains aggregate functions
        UsesEphemeral = 0x0008,     // Uses the OpenEphemeral opcode
        Expanded = 0x0010,		    // sqlite3SelectExpand() called on this
        HasTypeInfo = 0x0020,	    // FROM subqueries have Table metadata
        UseSorter = 0x0040,		    // Sort using a sorter
        Values = 0x0080,			// Synthesized from VALUES clause
        Materialize = 0x0100,	    // Force materialization of views
        NestedFrom = 0x0200,		// Part of a parenthesized FROM clause
    }

    public enum SRT : byte
    {
        Union = 1,				// Store result as keys in an index
        Except = 2,				// Remove result from a UNION index
        Exists = 3,				// Store 1 if the result is not empty
        Discard = 4,			// Do not save the results anywhere
        // IgnorableOrderby(x) : The ORDER BY clause is ignored for all of the above
        Output = 5,             // Output each row of result
        Mem = 6,                // Store result in a memory cell
        Set = 7,                // Store results as keys in an index
        Table = 8,              // Store result as data with an automatic rowid
        EphemTab = 9,           // Create transient tab and store like SRT_Table
        Coroutine = 10,         // Generate a single row of result
    }

    public class SelectDest
    {
        public SRT Dest;			// How to dispose of the results.  On of SRT_* above.
        public AFF AffSdst;		    // Affinity used when eDest==SRT_Set
        public int SDParmId;		// A parameter used by the eDest disposal method
        public int SdstId;			// Base register where results are written
        public int Sdsts;			// Number of registers allocated
    }

    public class Select
    {
        public ExprList EList;			// The fields of the result
        public TK OP;					// One of: TK_UNION TK_ALL TK_INTERSECT TK_EXCEPT
        public SF SelFlags;				// Various SF_* values
        public int LimitId, OffsetId;	// Memory registers holding LIMIT & OFFSET counters
        public OP[] AddrOpenEphms = new OP[3];		// OP_OpenEphem opcodes related to this select
        public double SelectRows;		// Estimated number of result rows
        public SrcList Src;				// The FROM clause
        public Expr Where;				// The WHERE clause
        public ExprList GroupBy;		// The GROUP BY clause
        public Expr Having;				// The HAVING clause
        public ExprList OrderBy;		// The ORDER BY clause
        public Select Prior;			// Prior select in a compound select statement
        public Select Next;				// Next select to the left in a compound
        public Select Rightmost;		// Right-most select in a compound select statement
        public Expr Limit;				// LIMIT expression. NULL means not used.
        public Expr Offset;				// OFFSET expression. NULL means not used.
    }

    public static partial class E
    {
        public static bool IgnorableOrderby(this SelectDest x) { return (x.Dest <= SRT.Discard); }
    }

    #endregion

    #region Expr

    public enum EP : ushort
    {
        FromJoin = 0x0001,		// Originated in ON or USING clause of a join
        Agg = 0x0002,			// Contains one or more aggregate functions
        Resolved = 0x0004,		// IDs have been resolved to COLUMNs
        Error = 0x0008,			// Expression contains one or more errors
        Distinct = 0x0010,		// Aggregate function with DISTINCT keyword
        VarSelect = 0x0020,		// pSelect is correlated, not constant
        DblQuoted = 0x0040,		// token.z was originally in "..."
        InfixFunc = 0x0080,		// True for an infix function: LIKE, GLOB, etc
        Collate = 0x0100,		// Tree contains a TK_COLLATE opeartor
        FixedDest = 0x0200,		// Result needed in a specific register
        IntValue = 0x0400,		// Integer value contained in u.iValue
        xIsSelect = 0x0800,		// x.pSelect is valid (otherwise x.pList is)
        Hint = 0x1000,			// Not used
        Reduced = 0x2000,		// Expr struct is EXPR_REDUCEDSIZE bytes only
        TokenOnly = 0x4000,		// Expr struct is EXPR_TOKENONLYSIZE bytes only
        Static = 0x8000,		// Held in memory not obtained from malloc()
    }

    public enum EP2 : byte
    {
        MallocedToken = 0x0001,	    // Need to sqlite3DbFree() Expr.zToken
        Irreducible = 0x0002,	    // Cannot EXPRDUP_REDUCE this Expr
    }

    public partial class Expr
    {
        public TK OP;				// Operation performed by this node
        public AFF Aff;				// The affinity of the column or 0 if not a column
        public EP Flags;			// Various flags.  EP_* See below
        public class _u
        {
            public string Token;	// Token value. Zero terminated and dequoted
            public int I;			// Non-negative integer value if EP_IntValue
        }
        public _u u;

        // If the EP_TokenOnly flag is set in the Expr.flags mask, then no space is allocated for the fields below this point. An attempt to
        // access them will result in a segfault or malfunction. 
        public Expr Left;			// Left subnode
        public Expr Right;			// Right subnode
        public class _x
        {
            public ExprList List;	// Function arguments or in "<expr> IN (<expr-list)"
            public Select Select;	// Used for sub-selects and "<expr> IN (<select>)"
        }
        public _x x;

        // If the EP_Reduced flag is set in the Expr.flags mask, then no space is allocated for the fields below this point. An attempt to
        // access them will result in a segfault or malfunction.
#if MAX_EXPR_DEPTH
        public int Height;					// Height of the tree headed by this node
#endif
        // TK_COLUMN: cursor number of table holding column
        // TK_REGISTER: register number
        // TK_TRIGGER: 1 -> new, 0 -> old
        public int TableId;
        // TK_COLUMN: column index.  -1 for rowid.
        // TK_VARIABLE: variable number (always >= 1).
        public yVars ColumnIdx;
        public short Agg;					// Which entry in pAggInfo->aCol[] or ->aFunc[]
        public short RightJoinTable;		// If EP_FromJoin, the right table of the join
        public EP2 Flags2;					// Second set of flags.  EP2_...
        // TK_REGISTER: original value of Expr.op
        // TK_COLUMN: the value of p5 for OP_Column
        // TK_AGG_FUNCTION: nesting depth
        public TK OP2;
        // Used by TK_AGG_COLUMN and TK_AGG_FUNCTION
        public AggInfo AggInfo;
        // Table for TK_COLUMN expressions.
        public Table Table;

        internal Expr memcpy(int flag = -1)
        {
            Expr p = new Expr();
            p.OP = OP;
            p.Aff = Aff;
            p.Flags = Flags;
            p.u = u;
            if (flag == E.EXPR_TOKENONLYSIZE) return p;
            if (Left != null) p.Left = Left.memcpy();
            if (Right != null) p.Right = Right.memcpy();
            p.x = x;
            if (flag == E.EXPR_REDUCEDSIZE) return p;
#if MAX_EXPR_DEPTH
            p.Height = Height;
#endif
            p.TableId = TableId;
            p.ColumnIdx = ColumnIdx;
            p.Agg = Agg;
            p.RightJoinTable = RightJoinTable;
            p.Flags2 = Flags2;
            p.OP2 = OP2;
            p.AggInfo = AggInfo;
            p.Table = Table;
            return p;
        }
    }

    public static partial class E
    {
#if DEBUG
        public static void ExprSetIrreducible(Expr x) { x.Flags2 |= EP2.Irreducible; }
#else
        public static void ExprSetIrreducible(Expr x) { }
#endif

        public static bool ExprHasProperty(Expr e, EP p) { return (e.Flags & p) == p; }
        public static bool ExprHasAnyProperty(Expr e, EP p) { return (e.Flags & p) != 0; }
        public static void ExprSetProperty(Expr e, EP p) { e.Flags |= p; }
        public static void ExprClearProperty(Expr e, EP p) { e.Flags &= ~p; }

        public const int EXPR_FULLSIZE = -1; //sizeof(Expr)           // Full size
        public const int EXPR_REDUCEDSIZE = -1; //offsetof(Expr, TableId)  // Common features
        public const int EXPR_TOKENONLYSIZE = -1; //offsetof(Expr, Left)   // Fewer features
        public const int EXPRDUP_REDUCE = 0x0001;  // Used reduced-size Expr nodes
    }

    public class ExprList
    {
        public class ExprListItem
        {
            public Expr Expr;			// The list of expressions
            public string Name;			// Token associated with this expression
            public string Span;			// Original text of the expression
            public SO SortOrder;        // 1 for DESC or 0 for ASC
            public bool Done;		    // A flag to indicate when processing is finished
            public bool SpanIsTab;	    // zSpan holds DB.TABLE.COLUMN
            public ushort OrderByCol;   // For ORDER BY, column number in result set
            public ushort Alias;        // Index into Parse.aAlias[] for zName
        }
        public int Exprs;				// Number of expressions on the list
        public int ECursor;				// VDBE Cursor associated with this ExprList
        // For each expression in the list
        public ExprListItem[] Ids;		// Alloc a power of two greater or equal to nExpr
    }

    public class ExprSpan
    {
        public Expr Expr;       // The expression parse tree
        public string Start;    // First character of input text
        public string End;      // One character past the end of input text
    }

    #endregion

    #region Walker

    public enum WRC : byte
    {
        Continue = 0,	// Continue down into children
        Prune = 1,		// Omit children but continue walking siblings
        Abort = 2,		// Abandon the tree walk
    }

    public partial class Walker
    {
        public Func<Walker, Expr, WRC> ExprCallback;       // Callback for expressions
        public Func<Walker, Select, WRC> SelectCallback;   // Callback for SELECTs
        public Parse Parse;                 // Parser context.
        public int WalkerDepth;				// Number of subqueries
        public class _u
        {
            public NameContext NC;          // Naming context
            public int I;                   // Integer value
            public SrcList SrcList;         // FROM clause
            public Expr.SrcCount SrcCount;	// Counting column references
        }
        public _u u; // Extra data for callback
    }

    #endregion

    #region Callback
    #endregion

    #region Prepare

    public class InitData
    {
        public Context Ctx; // The database being initialized
        public string ErrMsg; // Error message stored here
        public int Db; // 0 for main database.  1 for TEMP, 2.. for ATTACHed
        public RC RC; // Result code stored here
    }

    public partial class Prepare
    {
    }

    #endregion

    #region Parse

    public static partial class E
    {
#if OMIT_VIRTUALTABLE
        public static bool INDECLARE_VTABLE(this Parse x) { return false; }
#else
        public static bool INDECLARE_VTABLE(this Parse x) { return x.DeclareVTable; }
#endif
    }

    public class AutoincInfo
    {
        public AutoincInfo Next;		// Next info block in a list of them all
        public Table Table;			    // Table this info block refers to
        public int DB;					// Index in sqlite3.aDb[] of database holding pTab
        public int RegCtr;				// Memory register holding the rowid counter
    }

    public enum IN_INDEX : byte
    {
        ROWID = 1,
        EPH = 2,
        INDEX_ASC = 3,
        INDEX_DESC = 4,
    }

    public static partial class E
    {
#if OMIT_TRIGGER
        public static Parse Parse_Toplevel(this Parse p) { return p; }
#else
        public static Parse Parse_Toplevel(this Parse p) { return (p.Toplevel != null ? p.Toplevel : p); }
#endif
    }

    public partial class Parse
    {
        public class ColCache
        {
            public int Table;				// Table cursor number
            public int Column;				// Table column number
            public byte TempReg;			// iReg is a temp register that needs to be freed
            public int Level;				// Nesting level
            public int Reg;				    // Reg with value of this column. 0 means none.
            public int Lru;				    // Least recently used entry has the smallest value
        }

        public Context Ctx;				    // The main database structure
        public string ErrMsg;				// An error message
        public Vdbe V;					    // An engine for executing database bytecode
        public RC RC;						// Return code from execution
        public byte ColNamesSet;			// TRUE after OP_ColumnName has been issued to pVdbe
        public byte CheckSchema;			// Causes schema cookie check after an error
        public byte Nested;				    // Number of nested calls to the parser/code generator
        //byte TempReg;			            // Number of temporary registers in aTempReg[]
        public byte TempRegsInUse;		    // Number of aTempReg[] currently checked out
        //byte ColCaches;			        // Number of entries in aColCache[]
        public byte ColCacheIdx;			// Next entry in aColCache[] to replace
        public byte IsMultiWrite;			// True if statement may modify/insert multiple rows
        public bool MayAbort;				// True if statement may throw an ABORT exception
        public array_t3<byte, int> TempReg = new array_t3<byte, yVars>(new int[3]); // Holding area for temporary registers
        public int RangeRegs;				// Size of the temporary register block
        public int RangeRegIdx;			    // First register in temporary register block
        public int Errs;					// Number of errors seen
        public int Tabs;					// Number of previously allocated VDBE cursors
        public int Mems;					// Number of memory cells used so far
        public int Sets;					// Number of sets used so far
        public int Onces;					// Number of OP_Once instructions so far
        public int CkBase;					// Base register of data during check constraints
        public int CacheLevel;				// ColCache valid when aColCache[].iLevel<=iCacheLevel
        public int CacheCnt;				// Counter used to generate aColCache[].lru values
        public array_t3<byte, ColCache> ColCaches = new array_t3<byte, ColCache>(new ColCache[L.N_COLCACHE]); // One for each column cache entry
        public yDbMask WriteMask;			// Start a write transaction on these databases
        public yDbMask CookieMask;			// Bitmask of schema verified databases
        public int CookieGoto;				// Address of OP_Goto to cookie verifier subroutine
        public int[] CookieValue = new int[L.MAX_ATTACHED + 2];  // Values of cookies to verify
        public int RegRowid;				// Register holding rowid of CREATE TABLE entry
        public int RegRoot;				    // Register holding root page number for new objects
        public int MaxArgs;				    // Max args passed to user function by sub-program
        public Token ConstraintName;		// Name of the constraint currently being parsed
#if !OMIT_SHARED_CACHE
        // int TableLocks;			        // Number of locks in aTableLock
        public array_t<TableLock> TableLocks; // Required table locks for shared-cache mode
#endif
        public AutoincInfo Ainc;			// Information about AUTOINCREMENT counters

        // Information used while coding trigger programs.
        public Parse Toplevel;			    // Parse structure for main program (or NULL)
        public Table TriggerTab;			// Table triggers are being coded for
        public double QueryLoops;			// Estimated number of iterations of a query
        public uint Oldmask;				// Mask of old.* columns referenced
        public uint Newmask;				// Mask of new.* columns referenced
        public TK TriggerOp;			    // TK_UPDATE, TK_INSERT or TK_DELETE
        public OE Orconf;				    // Default ON CONFLICT policy for trigger steps
        public bool DisableTriggers;		// True to disable triggers

        // Above is constant between recursions.  Below is reset before and after each recursion
        public int VarsSeen;				// Number of '?' variables seen in the SQL so far
        //int nzVar;				        // Number of available slots in azVar[]
        public byte Explain;				// True if the EXPLAIN flag is found on the query
#if !OMIT_VIRTUALTABLE
        public bool DeclareVTable;			// True if inside sqlite3_declare_vtab()
        //int nVtabLock;			        // Number of virtual tables to lock
#endif
        //int nAlias;				        // Number of aliased result set columns
        public int Height;					// Expression tree height of current sub-select
#if !OMIT_EXPLAIN
        public int SelectId;				// ID of current select for EXPLAIN output
        public int NextSelectId;			// Next available select ID for EXPLAIN output
#endif
        public array_t<string> Vars;		// Pointers to names of parameters
        public Vdbe Reprepare;		        // VM being reprepared (sqlite3Reprepare())
        public array_t<int> Alias;			// Register used to hold aliased result
        public StringBuilder Tail;			// All SQL text past the last semicolon parsed
        public Table NewTable;			    // A table being constructed by CREATE TABLE
        public Trigger NewTrigger;		    // Trigger under construct by a CREATE TRIGGER
        public string AuthContext;	        // The 6th parameter to db->xAuth callbacks
        public Token NameToken;			    // Token with unqualified schema object name
        public Token LastToken;			    // The last token parsed
#if !OMIT_VIRTUALTABLE
        public Token Arg;					// Complete text of a module argument
        public array_t<Table> VTableLocks; // Pointer to virtual tables needing locking
#endif
        public Table ZombieTab;			// List of Table objects to delete after code gen
        public TriggerPrg TriggerPrg;		// Linked list of coded triggers
    }

    public struct AuthContext
    {
        public string AuthCtx;		    // Put saved Parse.zAuthContext here
        public Parse Parse;				// The Parse structure
    }

    #endregion

    //include "Context.cu.h"

    #region Table

    public enum COLFLAG : ushort
    {
        PRIMKEY = 0x0001,		            // Column is part of the primary key
        HIDDEN = 0x0002,		            // A hidden column in a virtual table
    }

    public class Column
    {
        public string Name;					// Name of this column
        public Expr Dflt;					// Default value of this column
        public string DfltName;				// Original text of the default value
        public string Type;					// Data type for this column
        public string Coll;					// Collating sequence.  If NULL, use the default
        public OE NotNull;				    // An OE_ code for handling a NOT NULL constraint
        public AFF Affinity;				// One of the SQLITE_AFF_... values
        public COLFLAG ColFlags;			// Boolean properties.  See COLFLAG_ defines below
    }

    public enum VTABLECONFIG : byte
    {
        CONSTRAINT = 1,
    }

    public partial class VTable
    {
        public Context Ctx;				    // Database connection associated with this table
        public TableModule Module;		    // Pointer to module implementation
        public IVTable IVTable;			    // Pointer to vtab instance
        public int Refs;					// Number of pointers to this structure
        public bool Constraint;			    // True if constraints are supported
        public int Savepoints;				// Depth of the SAVEPOINT stack
        public VTable Next;				    // Next in linked list (see above)

        public static bool InSync(Context ctx) { return (ctx.VTrans.length > 0 && ctx.VTrans.data == null); }
    }

    public enum TF : byte
    {
        Readonly = 0x01,        // Read-only system table
        Ephemeral = 0x02,       // An ephemeral table
        HasPrimaryKey = 0x04,   // Table has a primary key
        Autoincrement = 0x08,   // Integer primary key is autoincrement
        Virtual = 0x10,         // Is a virtual table
    }

    public partial class Table
    {
        public string Name;				    // Name of the table or view
        public array_t2<short, Column> Cols;    // Information about each column
        public Index Index;			        // List of SQL indexes on this table.
        public Select Select;			    // NULL for tables.  Points to definition if a view.
        public FKey FKeys;			        // Linked list of all foreign keys in this table
        public string ColAff;			    // String defining the affinity of each column
#if !OMIT_CHECK
        public ExprList Check;		        // All CHECK constraints
#endif
        public tRowcnt RowEst;			    // Estimated rows in table - from sqlite_stat1 table
        public int Id;				        // Root BTree node for this table (see note above)
        public short PKey;				    // If not negative, use aCol[iPKey] as the primary key
        public ushort Refs;                 // Number of pointers to this Table
        public TF TabFlags;			        // Mask of TF_* values
        public OE KeyConf;                  // What to do in case of uniqueness conflict on iPKey
#if !OMIT_ALTERTABLE
        public int AddColOffset;		    // Offset in CREATE TABLE stmt to add a new column
#endif
#if !OMIT_VIRTUALTABLE
        public array_t<string> ModuleArgs;  // Text of all module args. [0] is module name
        public VTable VTables;		        // List of VTable objects.
#endif
        public Trigger Triggers;		    // List of triggers stored in pSchema
        public Schema Schema;			    // Schema that contains this table
        public Table NextZombie;		    // Next on the Parse.pZombieTab list
    }

    public static partial class E
    {
#if !OMIT_VIRTUALTABLE
        public static bool IsVirtual(this Table x) { return ((x.TabFlags & TF.Virtual) != 0); }
        public static bool IsHiddenColumn(this Column x) { return ((x.ColFlags & COLFLAG.HIDDEN) != 0); }
#else
        public static bool IsVirtual(this Table x) { return false; }
        public static bool IsHiddenColumn(this Column x) { return false; }
#endif
    }

    public class FKey
    {
        public class ColMap                 // Mapping of columns in pFrom to columns in zTo
        {
            public int From;			    // Index of column in pFrom
            public string Col;			    // Name of column in zTo.  If 0 use PRIMARY KEY
        }
        public Table From;			        // Table containing the REFERENCES clause (aka: Child)
        public FKey NextFrom;			    // Next foreign key in pFrom
        public string To;				    // Name of table that the key points to (aka: Parent)
        public FKey NextTo;			        // Next foreign key on table named zTo
        public FKey PrevTo;			        // Previous foreign key on table named zTo
        public bool IsDeferred;		        // True if constraint checking is deferred till COMMIT
        public OE[] Actions = new OE[2];    // ON DELETE and ON UPDATE actions, respectively
        public Trigger[] Triggers = new Trigger[2];	// Triggers for aAction[] actions
        public array_t3<int, ColMap> Cols = new array_t3<yVars, ColMap>(new ColMap[1]); // One entry for each of nCol column s
    }

    #endregion

    #region Rowset
    #endregion

    #region Backup
    #endregion

    #region Authorization

    public enum ARC : byte
    {
        OK = 0,         // Successful result
        DENY = 1,       // Abort the SQL statement with an error
        IGNORE = 2,     // Don't allow access, but don't generate an error
    }

    public enum AUTH : byte
    {
        CREATE_INDEX = 1,           // Index Name      Table Name
        CREATE_TABLE = 2,           // Table Name      NULL
        CREATE_TEMP_INDEX = 3,      // Index Name      Table Name
        CREATE_TEMP_TABLE = 4,      // Table Name      NULL
        CREATE_TEMP_TRIGGER = 5,    // Trigger Name    Table Name
        CREATE_TEMP_VIEW = 6,       // View Name       NULL
        CREATE_TRIGGER = 7,         // Trigger Name    Table Name
        CREATE_VIEW = 8,            // View Name       NULL
        DELETE = 9,                 // Table Name      NULL
        DROP_INDEX = 10,            // Index Name      Table Name
        DROP_TABLE = 11,            // Table Name      NULL
        DROP_TEMP_INDEX = 12,       // Index Name      Table Name
        DROP_TEMP_TABLE = 13,       // Table Name      NULL
        DROP_TEMP_TRIGGER = 14,     // Trigger Name    Table Name
        DROP_TEMP_VIEW = 15,        // View Name       NULL
        DROP_TRIGGER = 16,          // Trigger Name    Table Name
        DROP_VIEW = 17,             // View Name       NULL
        INSERT = 18,                // Table Name      NULL
        PRAGMA = 19,                // Pragma Name     1st arg or NULL
        READ = 20,                  // Table Name      Column Name
        SELECT = 21,                // NULL            NULL
        TRANSACTION = 22,           // Operation       NULL
        UPDATE = 23,                // Table Name      Column Name
        ATTACH = 24,                // Filename        NULL
        DETACH = 25,                // Database Name   NULL
        ALTER_TABLE = 26,           // Database Name   Table Name
        REINDEX = 27,               // Index Name      NULL
        ANALYZE = 28,               // Table Name      NULL
        CREATE_VTABLE = 29,         // Table Name      Module Name
        DROP_VTABLE = 30,           // Table Name      Module Name
        FUNCTION = 31,              // NULL            Function Name
        SAVEPOINT = 32,             // Operation       Savepoint Name 
        COPY = 0,                   // No longer used
    }

    #endregion

    #region Savepoint

    public class Savepoint
    {
        public string Name;			// Savepoint name (nul-terminated)
        public long DeferredCons;	// Number of deferred fk violations
        public Savepoint Next;		// Parent savepoint (if any)
    }

    #endregion

    //include "Vdbe.cu.h"

    #region Trigger

    public enum TRIGGER : byte
    {
        BEFORE = 1,
        AFTER = 2,
    }

    public class Trigger
    {
        public string Name;             // The name of the trigger
        public string Table;            // The table or view to which the trigger applies
        public TK OP;                   // One of TK_DELETE, TK_UPDATE, TK_INSERT
        public TRIGGER TRtm;            // One of TRIGGER_BEFORE, TRIGGER_AFTER
        public Expr When;               // The WHEN clause of the expression (may be NULL)
        public IdList Columns;          // If this is an UPDATE OF <column-list> trigger, the <column-list> is stored here
        public Schema Schema;           // Schema containing the trigger
        public Schema TabSchema;        // Schema containing the table
        public TriggerStep StepList;    // Link list of trigger program steps
        public Trigger Next;            // Next trigger associated with the table

        public Trigger memcpy()
        {
            if (this == null)
                return null;
            Trigger cp = (Trigger)MemberwiseClone();
            if (When != null) cp.When = When.memcpy();
            if (Columns != null) cp.Columns = Columns.memcpy();
            if (Schema != null) cp.Schema = Schema.memcpy();
            if (TabSchema != null) cp.TabSchema = TabSchema.memcpy();
            if (StepList != null) cp.StepList = StepList.memcpy();
            if (Next != null) cp.Next = Next.memcpy();
            return cp;
        }
    }

    public class TriggerStep
    {
        public TK OP;               // One of TK_DELETE, TK_UPDATE, TK_INSERT, TK_SELECT
        public OE Orconf;         // OE_Rollback etc.
        public Trigger Trig;        // The trigger that this step is a part of
        public Select Select;       // SELECT statment or RHS of INSERT INTO .. SELECT ...
        public Token Target = new Token(); // Target table for DELETE, UPDATE, INSERT
        public Expr Where;          // The WHERE clause for DELETE or UPDATE steps
        public ExprList ExprList;   // SET clause for UPDATE.  VALUES clause for INSERT
        public IdList IdList;       // Column names for INSERT
        public TriggerStep Next;    // Next in the link-list
        public TriggerStep Last;    // Last element in link-list. Valid for 1st elem only

        internal TriggerStep memcpy()
        {
            if (this == null)
                return null;
            TriggerStep cp = (TriggerStep)MemberwiseClone();
            return cp;
        }
    };


    public class TriggerPrg
    {
        public Trigger Trigger;		    // Trigger this program was coded from
        public TriggerPrg Next;		    // Next entry in Parse.pTriggerPrg list
        public Vdbe.SubProgram Program;	// Program implementing pTrigger/orconf
        public OE Orconf;               // Default ON CONFLICT policy
        public uint[] Colmasks = new uint[2];     // Masks of old.*, new.* columns accessed
    }

    #endregion

    #region Attach

    public partial class DbFixer
    {
        public Parse Parse; // The parsing context.  Error messages written here
        public string DB; // Make sure all objects are contained in this database
        public string Type; // Type of the container - used for error messages
        public Token Name; // Name of the container - used for error messages
    }

    #endregion
}