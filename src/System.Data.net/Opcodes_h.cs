/* Automatically generated.  Do not edit */
/* See the mkopcodeh.awk script for details */
namespace Core
{
    public enum OP
    {
        Goto = 1,
        Gosub = 2,
        Return = 3,
        Yield = 4,
        HaltIfNull = 5,
        Halt = 6,
        Integer = 7,
        Int64 = 8,
        Real = 130,   /* same as TK_FLOAT    */
        String8 = 94,   /* same as TK_STRING   */
        String = 9,
        Null = 10,
        Blob = 11,
        Variable = 12,
        Move = 13,
        Copy = 14,
        SCopy = 15,
        ResultRow = 16,
        Concat = 91,   /* same as TK_CONCAT   */
        Add = 86,   /* same as TK_PLUS     */
        Subtract = 87,   /* same as TK_MINUS    */
        Multiply = 88,   /* same as TK_STAR     */
        Divide = 89,   /* same as TK_SLASH    */
        Remainder = 90,   /* same as TK_REM      */
        CollSeq = 17,
        Function = 18,
        BitAnd = 82,   /* same as TK_BITAND   */
        BitOr = 83,   /* same as TK_BITOR    */
        ShiftLeft = 84,   /* same as TK_LSHIFT   */
        ShiftRight = 85,   /* same as TK_RSHIFT   */
        AddImm = 20,
        MustBeInt = 21,
        RealAffinity = 22,
        ToText = 141,   /* same as TK_TO_TEXT  */
        ToBlob = 142,   /* same as TK_TO_BLOB  */
        ToNumeric = 143,   /* same as TK_TO_NUMERIC*/
        ToInt = 144,   /* same as TK_TO_INT   */
        ToReal = 145,   /* same as TK_TO_REAL  */
        Eq = 76,   /* same as TK_EQ       */
        Ne = 75,   /* same as TK_NE       */
        Lt = 79,   /* same as TK_LT       */
        Le = 78,   /* same as TK_LE       */
        Gt = 77,   /* same as TK_GT       */
        Ge = 80,   /* same as TK_GE       */
        Permutation = 23,
        Compare = 24,
        Jump = 25,
        And = 69,   /* same as TK_AND      */
        Or = 68,   /* same as TK_OR       */
        Not = 19,   /* same as TK_NOT      */
        BitNot = 93,   /* same as TK_BITNOT   */
        Once = 26,
        If = 27,
        IfNot = 28,
        IsNull = 73,   /* same as TK_ISNULL   */
        NotNull = 74,   /* same as TK_NOTNULL  */
        Column = 29,
        Affinity = 30,
        MakeRecord = 31,
        Count = 32,
        Savepoint = 33,
        AutoCommit = 34,
        Transaction = 35,
        ReadCookie = 36,
        SetCookie = 37,
        VerifyCookie = 38,
        OpenRead = 39,
        OpenWrite = 40,
        OpenAutoindex = 41,
        OpenEphemeral = 42,
        SorterOpen = 43,
        OpenPseudo = 44,
        Close = 45,
        SeekLt = 46,
        SeekLe = 47,
        SeekGe = 48,
        SeekGt = 49,
        Seek = 50,
        NotFound = 51,
        Found = 52,
        IsUnique = 53,
        NotExists = 54,
        Sequence = 55,
        NewRowid = 56,
        Insert = 57,
        InsertInt = 58,
        Delete = 59,
        ResetCount = 60,
        SorterCompare = 61,
        SorterData = 62,
        RowKey = 63,
        RowData = 64,
        Rowid = 65,
        NullRow = 66,
        Last = 67,
        SorterSort = 70,
        Sort = 71,
        Rewind = 72,
        SorterNext = 81,
        Prev = 92,
        Next = 95,
        SorterInsert = 96,
        IdxInsert = 97,
        IdxDelete = 98,
        IdxRowid = 99,
        IdxLT = 100,
        IdxGE = 101,
        Destroy = 102,
        Clear = 103,
        CreateIndex = 104,
        CreateTable = 105,
        ParseSchema = 106,
        LoadAnalysis = 107,
        DropTable = 108,
        DropIndex = 109,
        DropTrigger = 110,
        IntegrityCk = 111,
        RowSetAdd = 112,
        RowSetRead = 113,
        RowSetTest = 114,
        Program = 115,
        Param = 116,
        FkCounter = 117,
        FkIfZero = 118,
        MemMax = 119,
        IfPos = 120,
        IfNeg = 121,
        IfZero = 122,
        AggStep = 123,
        AggFinal = 124,
        Checkpoint = 125,
        JournalMode = 126,
        Vacuum = 127,
        IncrVacuum = 128,
        Expire = 129,
        TableLock = 131,
        VBegin = 132,
        VCreate = 133,
        VDestroy = 134,
        VOpen = 135,
        VFilter = 136,
        VColumn = 137,
        VNext = 138,
        VRename = 139,
        VUpdate = 140,
        Pagecount = 146,
        MaxPgcnt = 147,
        Trace = 148,
        Noop = 149,
        Explain = 150,
    }

    // Properties such as "out2" or "jump" that are specified in comments following the "case" for each opcode in the vdbe.c
    // are encoded into bitvectors as follows:
    public enum OPFLG
    {
        JUMP = 0x0001,  /* jump:  P2 holds jmp target */
        OUT2_PRERELEASE = 0x0002,  /* out2-prerelease: */
        IN1 = 0x0004,  /* in1:   P1 is an input */
        IN2 = 0x0008,  /* in2:   P2 is an input */
        IN3 = 0x0010,  /* in3:   P3 is an input */
        OUT2 = 0x0020,  /* out2:  P2 is an output */
        OUT3 = 0x0040,  /* out3:  P3 is an output */
    }

    public partial class E
    {
        public static readonly byte[] g_opcodeProperty = new byte[] {
/*   0 */ 0x00, 0x01, 0x01, 0x04, 0x04, 0x10, 0x00, 0x02,
/*   8 */ 0x02, 0x02, 0x02, 0x02, 0x02, 0x00, 0x00, 0x24,
/*  16 */ 0x00, 0x00, 0x00, 0x24, 0x04, 0x05, 0x04, 0x00,
/*  24 */ 0x00, 0x01, 0x01, 0x05, 0x05, 0x00, 0x00, 0x00,
/*  32 */ 0x02, 0x00, 0x00, 0x00, 0x02, 0x10, 0x00, 0x00,
/*  40 */ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x11, 0x11,
/*  48 */ 0x11, 0x11, 0x08, 0x11, 0x11, 0x11, 0x11, 0x02,
/*  56 */ 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
/*  64 */ 0x00, 0x02, 0x00, 0x01, 0x4c, 0x4c, 0x01, 0x01,
/*  72 */ 0x01, 0x05, 0x05, 0x15, 0x15, 0x15, 0x15, 0x15,
/*  80 */ 0x15, 0x01, 0x4c, 0x4c, 0x4c, 0x4c, 0x4c, 0x4c,
/*  88 */ 0x4c, 0x4c, 0x4c, 0x4c, 0x01, 0x24, 0x02, 0x01,
/*  96 */ 0x08, 0x08, 0x00, 0x02, 0x01, 0x01, 0x02, 0x00,
/* 104 */ 0x02, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
/* 112 */ 0x0c, 0x45, 0x15, 0x01, 0x02, 0x00, 0x01, 0x08,
/* 120 */ 0x05, 0x05, 0x05, 0x00, 0x00, 0x00, 0x02, 0x00,
/* 128 */ 0x01, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
/* 136 */ 0x01, 0x00, 0x01, 0x00, 0x00, 0x04, 0x04, 0x04,
/* 144 */ 0x04, 0x04, 0x02, 0x02, 0x00, 0x00, 0x00 };
    }
}