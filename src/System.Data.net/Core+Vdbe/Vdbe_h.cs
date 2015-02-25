using FILE = System.IO.TextWriter;
using System;

#region Limits
#if !MAX_VARIABLE_NUMBER
using ynVar = System.Int16;
#else
using ynVar = System.Int32; 
#endif
#if MAX_ATTACHED
using yDbMask = System.Int64; 
#else
using yDbMask = System.Int32;
#endif
#endregion

namespace Core
{
    public class Blob { }

    public partial class Vdbe
    {
        const int COLNAME_NAME = 0;
        const int COLNAME_DECLTYPE = 1;
        const int COLNAME_DATABASE = 2;
        const int COLNAME_TABLE = 3;
        const int COLNAME_COLUMN = 4;
#if ENABLE_COLUMN_METADATA
        const int COLNAME_N = 5;     // Number of COLNAME_xxx symbols
#else
#if OMIT_DECLTYPE
        const int COLNAME_N = 1;     // Number of COLNAME_xxx symbols
#else
        const int COLNAME_N = 2;
#endif
#endif
        static int ADDR(int x) { return -1 - x; }

        public enum STMTSTATUS : byte
        {
            FULLSCAN_STEP = 1,
            SORT = 2,
            AUTOINDEX = 3,
        }

        public enum P4T : sbyte
        {
            NOTUSED = 0,        // The P4 parameter is not used
            DYNAMIC = -1,       // Pointer to a string obtained from sqliteMalloc=();
            STATIC = -2,        // Pointer to a static string
            COLLSEQ = -4,       // P4 is a pointer to a CollSeq structure
            FUNCDEF = -5,       // P4 is a pointer to a FuncDef structure
            KEYINFO = -6,       // P4 is a pointer to a KeyInfo structure
            VDBEFUNC = -7,      // P4 is a pointer to a VdbeFunc structure
            MEM = -8,           // P4 is a pointer to a Mem*    structure
            TRANSIENT = 0,      // P4 is a pointer to a transient string
            VTAB = 10,          // P4 is a pointer to an sqlite3_vtab structure
            MPRINTF = -11,      // P4 is a string obtained from sqlite3_mprintf=();
            REAL = -12,         // P4 is a 64-bit floating point value
            INT64 = -13,        // P4 is a 64-bit signed integer
            INT32 = -14,        // P4 is a 32-bit signed integer
            INTARRAY = -15,     // P4 is a vector of 32-bit integers
            SUBPROGRAM = -18,   // P4 is a pointer to a SubProgram structure
            ADVANCE = -19,	    // P4 is a pointer to BtreeNext() or BtreePrev()
            // When adding a P4 argument using P4_KEYINFO, a copy of the KeyInfo structure is made.  That copy is freed when the Vdbe is finalized.  But if the
            // argument is P4_KEYINFO_HANDOFF, the passed in pointer is used.  It still gets freed when the Vdbe is finalized so it still should be obtained
            // from a single sqliteMalloc().  But no copy is made and the calling function should *not* try to free the KeyInfo.
            KEYINFO_HANDOFF = -16,
            KEYINFO_STATIC = -17,
        }

        public enum OPFLAG : byte
        {
            NCHANGE = 0x01,	// Set to update db->nChange
            LASTROWID = 0x02,	// Set to update db->lastRowid
            ISUPDATE = 0x04,	// This OP_Insert is an sql UPDATE
            APPEND = 0x08,	// This is likely to be an append
            USESEEKRESULT = 0x10,	// Try to avoid a seek in BtreeInsert()
            CLEARCACHE = 0x20,	// Clear pseudo-table cache in OP_Column
            LENGTHARG = 0x40,	// OP_Column only used for length()
            TYPEOFARG = 0x80,	// OP_Column only used for typeof()
            BULKCSR = 0x01,	// OP_Open** used to open bulk cursor
            P2ISREG = 0x02,	// P2 to OP_Open** is a register number
            PERMUTE = 0x01,	// OP_Compare: use the permutation
        }

        public class VdbeOp
        {
            public delegate RC AdvanceDelegate_t(Btree.BtCursor cur, ref int res);

            public OP Opcode;               // What operation to perform
            public P4T P4Type;              // One of the P4_xxx constants for p4
            public OPFLG Opflags;           // Mask of the OPFLG_* flags in opcodes.h
            public byte P5;                 // Fifth parameter is an unsigned character
            public int P1;                  // First operand
            public int P2;                  // Second parameter (often the jump destination)
            public int P3;                  // The third parameter
            public class P4_t
            {
                public int I;               // Integer value if p4type==P4_INT32
                public object P;            // Generic pointer
                public string Z;            // Pointer to data for string (char array) types
                public long I64;            // Used when p4type is P4T_INT64
                public double Real;         // Used when p4type is P4T_REAL
                public FuncDef Func;        // Used when p4type is P4T_FUNCDEF
                public VdbeFunc VdbeFunc;   // Used when p4type is P4T_VDBEFUNC
                public CollSeq Coll;        // Used when p4type is P4T_COLLSEQ
                public Mem Mem;             // Used when p4type is P4T_MEM
                public VTable VTable;         // Used when p4type is P4T_VTAB
                public KeyInfo KeyInfo;     // Used when p4type is P4T_KEYINFO
                public int[] Is;            // Used when p4type is P4T_INTARRAY
                public SubProgram Program;  // Used when p4type is P4T_SUBPROGRAM
                public VdbeOp.AdvanceDelegate_t Advance;   // Used when p4type is P4T_ADVANCE                
            }
            public P4_t P4 = new P4_t();    // fourth parameter
#if DEBUG
            public string Comment;          // Comment to improve readability
#endif
#if VDBE_PROFILE
            public int Cnt;                 // Number of times this instruction was executed
            public ulong Cycles;            // Total time spend executing this instruction
#endif

            public void _memset()
            {
                Opcode = 0;
                P4Type = 0;
                P5 = 0;
                P1 = 0;
                P2 = 0;
                P3 = 0;
                P4 = new P4_t();
#if DEBUG
                Comment = null;
#endif
#if VDBE_PROFILE
                Cnt = 0;
                Cycles = 0;
#endif
            }
        }

        public class SubProgram
        {
            public array_t<VdbeOp> Ops;     // Array of opcodes for sub-program
            public int Mems;                // Number of memory cells required
            public int Csrs;                // Number of cursors required
            public int Onces;               // Number of OP_Once instructions
            public int Token;               // id that may be used to recursive triggers
            public SubProgram Next;         // Next sub-program already visited
        }

        public struct VdbeOpList
        {
            public OP Opcode;               // What operation to perform
            public int P1;                  // First operand
            public int P2;                  // Second parameter (often the jump destination)
            public int P3;                  // Third parameter
            public VdbeOpList(OP opcode, int p1, int p2, int p3)
            {
                Opcode = opcode;
                P1 = p1;
                P2 = p2;
                P3 = p3;
            }
        }

        //#if OMIT_FLOATING_POINT
        ////# define sqlite3VdbeMemSetDouble sqlite3VdbeMemSetInt64
        //#else
        //        //void sqlite3VdbeMemSetDouble(Mem*, double);
        //#endif

        //#if !OMIT_SHARED_CACHE && THREADSAFE
        //        //void Enter(Vdbe);
        //        //void Leave(Vdbe);
        //#else
        //    static void Enter( Vdbe p ) { }
        //    static void Leave( Vdbe p ) { }
        //#endif

        //#if !OMIT_FOREIGN_KEY
        //        //int sqlite3VdbeCheckFk(Vdbe *, int);
        //#else
        //        static int sqlite3VdbeCheckFk(Vdbe p, int i) { return 0; }
        //#endif

        //#if !OMIT_INCRBLOB
        //        //      int sqlite3VdbeMemExpandBlob(Mem);
        //#else
        //        static RC sqlite3VdbeMemExpandBlob(Mem x) { return RC.OK; }
        //#endif

        public Context Ctx;                 // The database connection that owns this statement
        public array_t<VdbeOp> Ops;         // Space to hold the virtual machine's program
        public array_t<Mem> Mems;           // The memory locations
        public Mem[] Args;                  // Arguments to currently executing user function
        public Mem[] ColNames;              // Column names to return
        public Mem[] ResultSet;             // Pointer to an array of results
        public int OpsAlloc;                // Number of slots allocated for aOp[]
        public int LabelsAlloc;             // Number of slots allocated in aLabel[]
        public array_t<int> Labels;         // Space to hold the labels
        public ushort ResColumns;           // Number of columns in one row of the result set
        public uint Magic;                  // Magic number for sanity checking
        public string ErrMsg;               // Error message written here
        public Vdbe Prev, Next;             // Linked list of VDBEs with the same Vdbe.db
        public array_t<VdbeCursor> Cursors; // One element of this array for each open cursor
        public array_t2<ynVar, Mem> Vars;   // Values for the OP_Variable opcode.
        public array_t2<ynVar, string> VarNames; // Name of variables
        public uint CacheCtr;               // VdbeCursor row cache generation counter
        public int PC;                      // The program counter
        public RC RC_;                      // Value to return
        public OE ErrorAction;              // Recovery action to do in case of an error
        public int MinWriteFileFormat;      // Minimum file format for writable database files
        public byte HasExplain;             // True if EXPLAIN present on SQL command
        public byte InVtabMethod;           // See comments above
        public bool ChangeCntOn;            // True to update the change-counter
        public bool Expired;                // True if the VM needs to be recompiled
        public bool RunOnlyOnce;            // Automatically expire on reset
        public bool UsesStmtJournal;        // True if uses a statement journal
        public bool ReadOnly;               // True for read-only statements
        public bool IsPrepareV2;            // True if prepared with prepare_v2()
        public bool DoingRerun;             // True if rerunning after an auto-reprepare
        public int Changes;                 // Number of db changes made since last reset
        public yDbMask BtreeMask;           // Bitmask of db.aDb[] entries referenced
        public yDbMask LockMask;            // Subset of btreeMask that requires a lock
        public int StatementID;             // Statement number (or 0 if has not opened stmt)
        public int[] Counters = new int[3]; // Counters used by sqlite3_stmt_status()
#if !OMIT_TRACE
        public long StartTime;              // Time when query started - used for profiling
#endif
        public long FkConstraints;          // Number of imm. FK constraints this VM
        public long StmtDefCons;            // Number of def. constraints when stmt started
        public string Sql_;                 // Text of the SQL statement that generated this
        public object FreeThis;             // Free this when deleting the vdbe
#if DEBUG
        public FILE Trace;                  // Write an execution trace here, if not NULL
#endif
#if ENABLE_TREE_EXPLAIN
        Explain _explain;		            // The explainer
        string _explainString;              // Explanation of data structures
#endif
        public VdbeFrame Frames;            // Parent frame
        public int FramesLength;            // Number of frames in Frames list
        public VdbeFrame DelFrames;         // List of frame objects to free on VM reset
        public uint Expmask;                // Binding to these vars invalidates VM
        public SubProgram Programs;         // Linked list of all sub-programs used by VM
        public array_t<byte> OnceFlags;     // Flags for OP_Once

        public Vdbe _memcpy() { return (Vdbe)MemberwiseClone(); }
        public void _memcpy(Vdbe ct)
        {
            ct.Ctx = Ctx;
            ct.Ops = Ops; //ct.Ops.length = Ops.length;
            ct.Mems = Mems; //ct.Mems.length = Mems.length;
            ct.Args = Args; //ct.Args.length = Args.length;
            ct.ColNames = ColNames;
            ct.ResultSet = ResultSet;
            ct.OpsAlloc = OpsAlloc;
            ct.LabelsAlloc = LabelsAlloc;
            ct.Labels = Labels; //ct.Labels.length = Labels.length;
            ct.ResColumns = ResColumns;
            ct.Magic = Magic;
            ct.ErrMsg = ErrMsg;
            ct.Prev = Prev; ct.Next = Next;
            ct.Cursors = Cursors; //ct.Cursors.length = Cursors.length;
            ct.Vars = Vars; ct.Vars.length = Vars.length;
            ct.VarNames = VarNames; ct.VarNames.length = VarNames.length;
            ct.CacheCtr = CacheCtr;
            ct.PC = PC;
            ct.RC_ = RC_;
            ct.ErrorAction = ErrorAction;
            ct.MinWriteFileFormat = MinWriteFileFormat;
            ct.HasExplain = HasExplain;
            ct.InVtabMethod = InVtabMethod;
            ct.ChangeCntOn = ChangeCntOn;
            ct.Expired = Expired;
            ct.RunOnlyOnce = RunOnlyOnce;
            ct.UsesStmtJournal = UsesStmtJournal;
            ct.ReadOnly = ReadOnly;
            ct.IsPrepareV2 = IsPrepareV2;
            ct.DoingRerun = DoingRerun;
            ct.Changes = Changes;
            ct.BtreeMask = BtreeMask;
            ct.LockMask = LockMask;
            ct.StatementID = StatementID;
            Counters.CopyTo(ct.Counters, 0);
#if !OMIT_TRACE
            ct.StartTime = StartTime;
#endif
            ct.FkConstraints = FkConstraints;
            ct.StmtDefCons = StmtDefCons;
            ct.Sql_ = Sql_;
            ct.FreeThis = FreeThis;
#if DEBUG
            ct.Trace = Trace;
#endif
#if ENABLE_TREE_EXPLAIN
            ct._explain = _explain;
            ct._explainString = _explainString;
#endif
            ct.Frames = Frames;
            ct.DelFrames = DelFrames;
            ct.Expmask = Expmask;
            ct.Programs = Programs;
            ct.OnceFlags = OnceFlags;
        }

        const uint VDBE_MAGIC_INIT = 0x26bceaa5;    // Building a VDBE program
        const uint VDBE_MAGIC_RUN = 0xbdf20da3;     // VDBE is ready to execute
        const uint VDBE_MAGIC_HALT = 0x519c2973;    // VDBE has completed execution
        const uint VDBE_MAGIC_DEAD = 0xb606c3c8;    // The VDBE has been deallocated
    }

    public partial class E
    {
#if !OMIT_INCRBLOB
        public static RC ExpandBlob(Mem P) { return ((P.Flags & MEM.Zero) != 0 ? Vdbe.MemExpandBlob(P) : 0); }
#else
        public static RC ExpandBlob(Mem P) { return RC.OK; }
#endif
        public static void VdbeMemRelease(Mem X) { if ((X.Flags & (MEM.Agg | MEM.Dyn | MEM.RowSet | MEM.Frame)) != 0) Vdbe.MemReleaseExternal(X); }
    }
}
