// vdbeint.h
#include "Core+Vdbe.cu.h"
#include "..\Core+Btree\BtreeInt.cu.h"
namespace Core
{
	struct VTableCursor;
	struct VdbeSorter;
	struct RowSet;

	struct VdbeCursor
	{
		BtCursor *Cursor;		// The cursor structure of the backend
		Btree *Bt;				// Separate file holding temporary table
		KeyInfo *KeyInfo;		// Info about index keys needed by index cursors
		int Db;					// Index of cursor database in db->aDb[] (or -1)
		int PseudoTableReg;		// Register holding pseudotable content.
		int Fields;				// Number of fields in the header
		bool Zeroed;			// True if zeroed out and ready for reuse
		bool RowidIsValid;		// True if lastRowid is valid
		bool AtFirst;			// True if pointing to first entry
		bool UseRandomRowid;	// Generate new record numbers semi-randomly
		bool NullRow;			// True if pointing to a row with no data
		bool DeferredMoveto;	// A call to sqlite3BtreeMoveto() is needed
		bool IsTable;			// True if a table requiring integer keys
		bool IsIndex;			// True if an index containing keys only - no data
		bool IsOrdered;			// True if the underlying table is BTREE_UNORDERED
		bool IsSorter;			// True if a new-style sorter
		bool MultiPseudo;		// Multi-register pseudo-cursor
#ifndef OMIT_VIRTUALTABLE
		IVTableCursor *VtabCursor;		// The cursor for a virtual table
		const ITableModule *IModule;    // Module for cursor pVtabCursor
#endif
		int64 SeqCount;			// Sequence counter
		int64 MovetoTarget;		// Argument to the deferred sqlite3BtreeMoveto()
		int64 LastRowid;		// Last rowid from a Next or NextIdx operation
		VdbeSorter *Sorter;		// Sorter object for OP_SorterOpen cursors

		// Result of last sqlite3BtreeMoveto() done by an OP_NotExists or OP_IsUnique opcode on this cursor.
		int SeekResult;

		// Cached information about the header for the data record that the cursor is currently pointing to.  Only valid if cacheStatus matches
		// Vdbe.cacheCtr.  Vdbe.cacheCtr will never take on the value of CACHE_STALE and so setting cacheStatus=CACHE_STALE guarantees that
		// the cache is out of date.
		//
		// aRow might point to (ephemeral) data for the current row, or it might be NULL.
		uint32 CacheStatus;     // Cache is valid if this matches Vdbe.cacheCtr
		int PayloadSize;		// Total number of bytes in the record
		uint32 *Types;          // Type values for all entries in the record
		uint32 *Offsets;        // Cached offsets to the start of each columns data
		uint8 *Rows;            // Data for the current row, if all on one page
	};

	struct VdbeFrame
	{
		Vdbe *V;					// VM this frame belongs to
		struct VdbeFrame *Parent;	// Parent of this frame, or NULL if parent is main
		array_t<Vdbe::VdbeOp> Ops;	// Program instructions for parent frame
		array_t<Mem> Mems;			// Array of memory cells for parent frame
		array_t<uint8> OnceFlags;	// Array of OP_Once flags for parent frame
		array_t<VdbeCursor*> Cursors;	// Array of Vdbe cursors for parent frame
		void *Token;				// Copy of SubProgram.token
		int64 LastRowID;			// Last insert rowid (sqlite3.lastRowid)
		int PC;						// Program Counter in parent (calling) frame
		int ChildMems;				// Number of memory cells for child frame
		int ChildCursors;			// Number of cursors for child frame
		int Changes;				// Statement changes (Vdbe.nChanges)
	};

#define VdbeFrameMem(p) ((Mem *)&((uint8 *)p)[_ROUND8(sizeof(VdbeFrame))])

#define CACHE_STALE 0 // A value for VdbeCursor.cacheValid that means the cache is always invalid.

	enum MEM : uint16
	{
		// If the MEM_Null flag is set, then the value is an SQL NULL value. No other flags may be set in this case.
		//
		// If the MEM_Str flag is set then Mem.z points at a string representation. Usually this is encoded in the same unicode encoding as the main
		// database (see below for exceptions). If the MEM_Term flag is also set, then the string is nul terminated. The MEM_Int and MEM_Real 
		// flags may coexist with the MEM_Str flag.
		MEM_Null = 0x0001,		// Value is NULL
		MEM_Str = 0x0002,		// Value is a string
		MEM_Int = 0x0004,		// Value is an integer
		MEM_Real = 0x0008,		// Value is a real number
		MEM_Blob = 0x0010,		// Value is a BLOB
		MEM_RowSet = 0x0020,	// Value is a RowSet object
		MEM_Frame = 0x0040,		// Value is a VdbeFrame object
		MEM_Invalid = 0x0080,	// Value is undefined
		MEM_Cleared = 0x0100,	// NULL set by OP_Null, not from data
		MEM_TypeMask = 0x01ff,  // Mask of type bits
		// Whenever Mem contains a valid string or blob representation, one of the following flags must be set to determine the memory management
		// policy for Mem.z.  The MEM_Term flag tells us whether or not the string is \000 or \u0000 terminated
		MEM_Term = 0x0200,		// String rep is nul terminated
		MEM_Dyn = 0x0400,		// Need to call sqliteFree() on Mem.z
		MEM_Static = 0x0800,	// Mem.z points to a static string
		MEM_Ephem = 0x1000,		// Mem.z points to an ephemeral string
		MEM_Agg = 0x2000,		// Mem.z points to an agg function context
#ifndef OMIT_INCRBLOB
		MEM_Zero = 0x4000,		// Mem.i contains count of 0s appended to blob
#else
		MEM_Zero = 0x0000,
#endif
	};
	__device__ inline void operator|=(MEM &a, int b) { a = (MEM)(a | b); }
	__device__ inline void operator&=(MEM &a, int b) { a = (MEM)(a & b); }
	__device__ inline MEM operator|(MEM a, MEM b) { return (MEM)((int)a | (int)b); }
	__device__ inline MEM operator&(MEM a, MEM b) { return (MEM)((int)a & (int)b); }

#define MemSetTypeFlag(p, f) ((p)->Flags = (MEM)(((p)->Flags&~(MEM_TypeMask|MEM_Zero))|(int)f))
#ifdef _DEBUG
#define MemIsValid(M) ((M)->Flags & MEM_Invalid)==0
#endif

	struct Mem
	{
		Context *Ctx;			// The associated database connection
		char *Z;				// String or BLOB value
		double R;				// Real value
		union {
			int64 I;            // Integer value used when MEM_Int is set in flags
			int Zeros;			// Used when bit MEM_Zero is set in flags
			FuncDef *Def;		// Used only when flags==MEM_Agg
			RowSet *RowSet;		// Used only when flags==MEM_RowSet
			VdbeFrame *Frame;	// Used when flags==MEM_Frame
		} u;
		int N;					// Number of characters in string value, excluding '\0'
		MEM Flags;				// Some combination of MEM_Null, MEM_Str, MEM_Dyn, etc.
		TYPE Type;				// One of SQLITE_NULL, SQLITE_TEXT, SQLITE_INTEGER, etc
		TEXTENCODE Encode;		// SQLITE_UTF8, SQLITE_UTF16BE, SQLITE_UTF16LE
#ifdef _DEBUG
		Mem *ScopyFrom;			// This Mem is a shallow copy of pScopyFrom
		void *Filler;			// So that sizeof(Mem) is a multiple of 8
#endif
		void (*Del)(void *);	// If not null, call this function to delete Mem.z
		char *Malloc;			// Dynamic buffer allocated by sqlite3_malloc()

		//		inline const void *GetBlob()
		//		{
		//			if (Flags & (MEM_Blob|MEM_Str))
		//			{
		//				sqlite3VdbeMemExpandBlob(p);
		//				Flags &= ~MEM_Str;
		//				Flags |= MEM_Blob;
		//				return (N ? Z : nullptr);
		//			}
		//			return GetText();
		//		}
		//		inline int GetBytes() { return ValueBytes(this, TEXTENCODE_UTF8); }
		//		inline int GetBytes16() { return ValueBytes(this, TEXTENCODE_UTF16NATIVE); }
		//		inline double GetDouble() { return RealValue(this); }
		//		inline int GetInt() { return (int)IntValue(this); }
		//		inline int64 GetInt64(){ return IntValue(this); }
		//		inline const unsigned char *GetText() { return (const unsigned char *)ValueText(this, TEXTENCODE_UTF8); }
		//#ifndef OMIT_UTF16
		//		inline const void *GetText16() { return ValueText(this, TEXTENCODE_UTF16NATIVE); }
		//		inline const void *GetText16be() { return Vdbe::Value_Text(this, TEXTENCODE_UTF16BE); }
		//		inline const void *GetText16le() { return Vdbe::Value_Text(this, TEXTENCODE_UTF16LE); }
		//#endif
		//		inline TYPE GetType() { return Type; }
	};

	struct VdbeFunc // : public FuncDef
	{
		FuncDef *Func;			// The definition of the function
		int AuxsLength;         // Number of entries allocated for apAux[]
		struct AuxData
		{
			void *Aux;                 // Aux data for the i-th argument
			void (*Delete)(void *);    // Destructor for the aux data
		} Auxs[1];              // One slot for each function argument
	};

	// was:sqlite3_context
	struct FuncContext
	{
		FuncDef *Func;			// Pointer to function information.  MUST BE FIRST
		VdbeFunc *VdbeFunc;		// Auxilary data, if created.
		Mem S;					// The return value is stored here
		Mem *Mem;				// Memory cell used to store aggregate context
		CollSeq *Coll;			// Collating sequence
		RC IsError;				// Error code returned by the function
		bool SkipFlag;			// Skip skip accumulator loading if true
	};

	struct Explain
	{
		Vdbe *Vdbe_;			// Attach the explanation to this Vdbe
		TextBuilder Str;		// The string being accumulated
		int IndentLength;		// Number of elements in aIndent
		uint16 Indents[100];	// Levels of indentation
		char ZBase[100];		// Initial space
	};

#define VDBE_MAGIC_INIT     0x26bceaa5    // Building a VDBE program
#define VDBE_MAGIC_RUN      0xbdf20da3    // VDBE is ready to execute
#define VDBE_MAGIC_HALT     0x519c2973    // VDBE has completed execution
#define VDBE_MAGIC_DEAD     0xb606c3c8    // The VDBE has been deallocated

#define MASTER_NAME       "sqlite_master" // Name of the master database table.  The master database table is a special table that holds the names and attributes of all user tables and indices.
#define TEMP_MASTER_NAME  "sqlite_temp_master" // Name of the master database table.  The master database table is a special table that holds the names and attributes of all user tables and indices.
#define SCHEMA_TABLE(x)  (!E_OMIT_TEMPDB && (x) == 1 ? TEMP_MASTER_NAME : MASTER_NAME) // The name of the schema table.

}