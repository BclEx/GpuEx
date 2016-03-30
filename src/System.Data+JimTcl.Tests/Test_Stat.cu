// This file contains an implementation of the "dbstat" virtual table.
//
// The dbstat virtual table is used to extract low-level formatting information from an SQLite database in order to implement the
// "sqlite3_analyzer" utility.  See the ../tool/spaceanal.tcl script for an example implementation.
#include <Core+Vdbe\Core+Vdbe.cu.h>

#ifndef OMIT_VIRTUALTABLE

// Page paths:
// 
//   The value of the 'path' column describes the path taken from the root-node of the b-tree structure to each page. The value of the 
//   root-node path is '/'.
//
//   The value of the path for the left-most child page of the root of a b-tree is '/000/'. (Btrees store content ordered from left to right
//   so the pages to the left have smaller keys than the pages to the right.) The next to left-most child of the root page is
//   '/001', and so on, each sibling page identified by a 3-digit hex value. The children of the 451st left-most sibling have paths such
//   as '/1c2/000/, '/1c2/001/' etc.
//
//   Overflow pages are specified by appending a '+' character and a six-digit hexadecimal value to the path to the cell they are linked
//   from. For example, the three overflow pages in a chain linked from the left-most cell of the 450th child of the root page are identified by the paths:
//
//      '/1c2/000+000000'         // First page in overflow chain
//      '/1c2/000+000001'         // Second page in overflow chain
//      '/1c2/000+000002'         // Third page in overflow chain
//
//   If the paths are sorted using the BINARY collation sequence, then the overflow pages associated with a cell will appear earlier in the sort-order than its child page:
//
//      '/1c2/000/'               // Left-most child of 451st child of root
#define VTAB_SCHEMA \
	"CREATE TABLE xx(" \
	"name       STRING,           /* Name of table or index */" \
	"path       INTEGER,          /* Path to page from root */" \
	"pageno     INTEGER,          /* Page number */" \
	"pagetype   STRING,           /* 'internal', 'leaf' or 'overflow' */" \
	"ncell      INTEGER,          /* Cells on page (0 for overflow) */" \
	"payload    INTEGER,          /* Bytes of payload on this page */" \
	"unused     INTEGER,          /* Bytes of unused space on this page */" \
	"mx_payload INTEGER,          /* Largest payload size of all cells */" \
	"pgoffset   INTEGER,          /* Offset of page in file */" \
	"pgsize     INTEGER           /* Size of the page */" \
	");"

typedef struct StatTable StatTable;
typedef struct StatCursor StatCursor;
typedef struct StatPage StatPage;
typedef struct StatCell StatCell;

struct StatCell
{
	int Local;						// Bytes of local payload
	uint32 ChildPg;					// Child node (or 0 if this is a leaf)
	array_t<uint32> Ovfls;          // Array of overflow page numbers
	int LastOvfl;					// Bytes of payload on final overflow page
	int OvflId;                     // Iterates through aOvfl[]
};

struct StatPage
{
	uint32 Pgno;
	IPage *Pg;
	int CellId;
	char *Path;                    // Path to this page
	// Variables populated by statDecodePage():
	uint8 Flags;                    // Copy of flags byte
	int Unused;						// Number of unused bytes on page
	array_t<StatCell>Cells;			// Array of parsed cells
	uint32 RightChildPg;			// Right-child page number (or 0)
	int MaxPayload;                 // Largest payload of any cell on this page
};

struct StatCursor
{
	IVTableCursor base;
	Vdbe *Stmt;						// Iterates through set of root pages
	bool IsEof;                     // After pStmt has returned SQLITE_DONE

	StatPage Pages[32];
	int PageId;                      // Current entry in aPage[]

	// Values to return
	char *Name;                    // Value of 'name' column
	char *Path;                    // Value of 'path' column
	uint32 Pageno;					// Value of 'pageno' column
	char *Pagetype;                // Value of 'pagetype' column
	int Cells;                      // Value of 'ncell' column
	int Payload;                   // Value of 'payload' column
	int Unused;                    // Value of 'unused' column
	int MaxPayload;                 // Value of 'mx_payload' column
	int64 Offset;                  // Value of 'pgOffset' column
	int SizePage;                   // Value of 'pgSize' column
};

struct StatTable
{
	IVTable base;
	Context *Ctx;
};

#ifndef get2byte
#define get2byte(x) ((x)[0]<<8 | (x)[1])
#endif

// Connect to or create a statvfs virtual table.
__device__ static RC statConnect(Context *ctx, void *aux, int argc, const char *const args[], IVTable **vtab, char **err)
{
	StatTable *tab = (StatTable *)_alloc(sizeof(StatTable));
	memset(tab, 0, sizeof(StatTable));
	tab->Ctx = ctx;
	VTable::DeclareVTable(ctx, VTAB_SCHEMA);
	*vtab = &tab->base;
	return RC_OK;
}

// Disconnect from or destroy a statvfs virtual table.
__device__ static RC statDisconnect(IVTable *vtab)
{
	_free(vtab);
	return RC_OK;
}

// There is no "best-index". This virtual table always does a linear scan of the binary VFS log file.
__device__ static RC statBestIndex(IVTable *tab, IIndexInfo *idxInfo)
{
	// Records are always returned in ascending order of (name, path). If this will satisfy the client, set the orderByConsumed flag so that SQLite does not do an external sort.
	if ((idxInfo->OrderBys.length == 1 && idxInfo->OrderBys[0].Column == 0 && !idxInfo->OrderBys[0].Desc) ||
		(idxInfo->OrderBys.length == 2 && idxInfo->OrderBys[0].Column == 0 && !idxInfo->OrderBys[0].Desc && idxInfo->OrderBys[1].Column == 1 && !idxInfo->OrderBys[1].Desc))
		idxInfo->OrderByConsumed = true;
	idxInfo->EstimatedCost = 10.0;
	return RC_OK;
}

// Open a new statvfs cursor.
__device__ static RC statOpen(IVTable *vtab, IVTableCursor **cursor)
{
	StatTable *tab = (StatTable *)vtab;
	StatCursor *cur = (StatCursor *)_alloc(sizeof(StatCursor));
	memset(cur, 0, sizeof(StatCursor));
	cur->base.IVTable = vtab;
	RC rc = Prepare::Prepare_v2(tab->Ctx, 
		"SELECT 'sqlite_master' AS name, 1 AS rootpage, 'table' AS type"
		"  UNION ALL  "
		"SELECT name, rootpage, type FROM sqlite_master WHERE rootpage!=0"
		"  ORDER BY name", -1,
		&cur->Stmt, nullptr);
	if (rc != RC_OK)
	{
		_free(cur);
		return rc;
	}
	*cursor = (IVTableCursor *)cur;
	return RC_OK;
}

__device__ static void statClearPage(StatPage *p)
{
	for (int i = 0; i < p->Cells.length; i++)
		_free(p->Cells[i].Ovfls.data);
	Pager::Unref(p->Pg);
	_free(p->Cells.data);
	_free(p->Path);
	_memset(p, 0, sizeof(StatPage));
}

__device__ static void statResetCsr(StatCursor *cur)
{
	cur->Stmt->Reset();
	for (int i = 0; i < _lengthof(cur->Pages); i++)
		statClearPage(&cur->Pages[i]);
	cur->PageId = 0;
	_free(cur->Path);
	cur->Path = nullptr;
}

// Close a statvfs cursor.
__device__ static RC statClose(IVTableCursor *cursor)
{
	StatCursor *cur = (StatCursor *)cursor;
	statResetCsr(cur);
	Vdbe::Finalize(cur->Stmt);
	_free(cur);
	return RC_OK;
}

__device__ static void getLocalPayload(int usable, uint8 flags, int total, int *localOut)
{
	int minLocal;
	int maxLocal;
	if (flags == 0x0D)
	{             
		// Table leaf node
		minLocal = (usable - 12) * 32 / 255 - 23;
		maxLocal = usable - 35;
	}
	else
	{                          
		// Index interior and leaf nodes
		minLocal = (usable - 12) * 32 / 255 - 23;
		maxLocal = (usable - 12) * 64 / 255 - 23;
	}
	int local = minLocal + (total - minLocal) % (usable - 4);
	if (local > maxLocal) local = minLocal;
	*localOut = local;
}

__device__ static RC statDecodePage(Btree *bt, StatPage *p)
{
	uint8 *data = (uint8 *)Pager::GetData(p->Pg);
	uint8 *header = &data[p->Pgno == 1 ? 100 : 0];
	p->Flags = header[0];
	p->Cells.length = get2byte(&header[3]);
	p->MaxPayload = 0;
	int isLeaf = (p->Flags == 0x0A || p->Flags == 0x0D);
	int headerLength = 12 - isLeaf*4 + (p->Pgno == 1)*100;
	int unused = get2byte(&header[5]) - headerLength - 2*p->Cells.length;
	unused += (int)header[7];
	int offset = get2byte(&header[1]);
	while (offset)
	{
		unused += get2byte(&data[offset+2]);
		offset = get2byte(&data[offset]);
	}
	p->Unused = unused;
	p->RightChildPg = (isLeaf ? 0 : _convert_get4(&header[8]));

	int sizePage = bt->GetPageSize();
	if (p->Cells.length)
	{
		int i; // Used to iterate through cells
		int usable = sizePage - bt->GetReserve();
		p->Cells.data = (StatCell *)_alloc((p->Cells.length+1) * sizeof(StatCell));
		_memset(p->Cells.data, 0, (p->Cells.length+1) * sizeof(StatCell));
		for (i = 0; i < p->Cells.length; i++)
		{
			StatCell *cell = &p->Cells[i];
			offset = get2byte(&data[headerLength+i*2]);
			if (!isLeaf)
			{
				cell->ChildPg = _convert_get4(&data[offset]);
				offset += 4;
			}
			if ( p->Flags == 0x05) { } // A table interior node. nPayload==0.
			else
			{
				uint32 payload;        // Bytes of payload total (local+overflow)
				int local;             // Bytes of payload stored locally
				offset += _convert_getvarint32(&data[offset], payload);
				if (p->Flags == 0x0D)
				{
					uint64 dummy;
					offset += _convert_getvarint(&data[offset], &dummy);
				}
				if (payload > (uint32)p->MaxPayload) p->MaxPayload = payload;
				getLocalPayload(usable, p->Flags, payload, &local);
				cell->Local = local;
				_assert(local >= 0);
				_assert(payload >= (uint32)local);
				_assert(local <= (usable-35));
				if (payload > (uint32)local)
				{
					int ovfl = ((payload - local) + usable-4 - 1) / (usable - 4);
					cell->LastOvfl = (payload-local) - (ovfl-1) * (usable-4);
					cell->Ovfls.length = ovfl;
					cell->Ovfls.data = (uint32 *)_alloc(sizeof(uint32)*ovfl);
					cell->Ovfls[0] = _convert_get4(&data[offset+local]);
					for (int j = 1; j < ovfl; j++)
					{
						uint32 prev = cell->Ovfls[j-1];
						IPage *pg = 0;
						RC rc = bt->get_Pager()->Acquire(prev, &pg);
						if (rc != RC_OK)
						{
							_assert(pg == nullptr);
							return rc;
						} 
						cell->Ovfls[j] = _convert_get4((uint8 *)Pager::GetData(pg));
						Pager::Unref(pg);
					}
				}
			}
		}
	}

	return RC_OK;
}

// Populate the cur->iOffset and cur->szPage member variables. Based on the current value of cur->iPageno.
__device__ static void statSizeAndOffset(StatCursor *cur)
{
	StatTable *tab = (StatTable *)((IVTableCursor *)cur)->IVTable;
	Btree *bt = tab->Ctx->DBs[0].Bt;
	Pager *pager = bt->get_Pager();

	// The default page size and offset
	cur->SizePage = bt->GetPageSize();
	cur->Offset = (int64)cur->SizePage * (cur->Pageno - 1);

	// If connected to a ZIPVFS backend, override the page size and offset with actual values obtained from ZIPVFS.
	VFile *fd = pager->get_File();
	int64 x[2];
	x[0] = cur->Pageno;
	if (fd->FileControl((VFile::FCNTL)230440, &x) == RC_OK)
	{
		cur->Offset = x[0];
		cur->SizePage = (int)x[1];
	}
}

// Move a statvfs cursor to the next entry in the file.
__device__ static RC statNext(IVTableCursor *cursor)
{
	StatCursor *cur = (StatCursor *)cursor;
	StatTable *tab = (StatTable *)cursor->IVTable;
	Btree *bt = tab->Ctx->DBs[0].Bt;
	Pager *pager = bt->get_Pager();
	_free(cur->Path);
	cur->Path = nullptr;
	RC rc;
	if (!cur->Pages[0].Pg)
	{
		rc = cur->Stmt->Step();
		if (rc == RC_ROW)
		{
			uint32 root = (uint32)Vdbe::Column_Int64(cur->Stmt, 1);
			uint32 pages;
			pager->Pages(&pages);
			if (pages == 0)
			{
				cur->IsEof = true;
				return cur->Stmt->Reset();
			}
			rc = pager->Acquire(root, &cur->Pages[0].Pg);
			cur->Pages[0].Pgno = root;
			cur->Pages[0].CellId = 0;
			cur->Pages[0].Path = _mprintf("/");
			cur->PageId = 0;
		}
		else
		{
			cur->IsEof = true;
			return cur->Stmt->Reset();
		}
	}
	else
	{
		// Page p itself has already been visited.
		StatPage *p = &cur->Pages[cur->PageId];
		while (p->CellId < p->Cells.length)
		{
			StatCell *cell = &p->Cells[p->CellId];
			if (cell->OvflId < cell->Ovfls.length)
			{
				int usable = bt->GetPageSize()-bt->GetReserve();
				cur->Name = (char *)Vdbe::Column_Text(cur->Stmt, 0);
				cur->Pageno = cell->Ovfls[cell->OvflId];
				cur->Pagetype = "overflow";
				cur->Cells = 0;
				cur->MaxPayload = 0;
				cur->Path = _mprintf("%s%.3x+%.6x", p->Path, p->CellId, cell->OvflId);
				if (cell->OvflId < cell->Ovfls.length-1)
				{
					cur->Unused = 0;
					cur->Payload = usable - 4;
				}
				else
				{
					cur->Payload = cell->LastOvfl;
					cur->Unused = usable - 4 - cur->Payload;
				}
				cell->OvflId++;
				statSizeAndOffset(cur);
				return RC_OK;
			}
			if (p->RightChildPg) break;
			p->CellId++;
		}

		while (!p->RightChildPg || p->CellId > p->Cells.length)
		{
			statClearPage(p);
			if (cur->PageId == 0) return statNext(cursor);
			cur->PageId--;
			p = &cur->Pages[cur->PageId];
		}
		cur->PageId++;
		_assert(p == &cur->Pages[cur->PageId-1]);

		p[1].Pgno = (p->CellId == p->Cells.length ? p->RightChildPg : p->Cells[p->CellId].ChildPg);
		rc = pager->Acquire(p[1].Pgno, &p[1].Pg);
		p[1].CellId = 0;
		p[1].Path = _mprintf("%s%.3x/", p->Path, p->CellId);
		p->CellId++;
	}

	// Populate the StatCursor fields with the values to be returned by the xColumn() and xRowid() methods.
	if (rc == RC_OK)
	{
		StatPage *p = &cur->Pages[cur->PageId];
		cur->Name = (char *)Vdbe::Column_Text(cur->Stmt, 0);
		cur->Pageno = p->Pgno;
		statDecodePage(bt, p);
		statSizeAndOffset(cur);
		switch (p->Flags)
		{
		case 0x05:             // table internal
		case 0x02:             // index internal
			cur->Pagetype = "internal";
			break;
		case 0x0D:             // table leaf
		case 0x0A:             // index leaf
			cur->Pagetype = "leaf";
			break;
		default:
			cur->Pagetype = "corrupted";
			break;
		}
		cur->Cells = p->Cells.length;
		cur->Unused = p->Unused;
		cur->MaxPayload = p->MaxPayload;
		cur->Path = _mprintf("%s", p->Path);
		int payload = 0;
		for (int i = 0; i < p->Cells.length; i++)
			payload += p->Cells[i].Local;
		cur->Payload = payload;
	}
	return rc;
}

__device__ static bool statEof(IVTableCursor *cursor)
{
	StatCursor *cur = (StatCursor *)cursor;
	return cur->IsEof;
}

__device__ static RC statFilter(IVTableCursor *cursor, int idxNum, const char *idxStr, int argc, Mem **args)
{
	StatCursor *cur = (StatCursor *)cursor;
	statResetCsr(cur);
	return statNext(cursor);
}

__device__ static RC statColumn(IVTableCursor *cursor,  FuncContext *fctx, int i)
{
	StatCursor *cur = (StatCursor *)cursor;
	switch (i)
	{
	case 0: Vdbe::Result_Text(fctx, cur->Name, -1, DESTRUCTOR_STATIC); break; // name
	case 1: Vdbe::Result_Text(fctx, cur->Path, -1, DESTRUCTOR_TRANSIENT); break; // path
	case 2: Vdbe::Result_Int64(fctx, cur->Pageno); break; // pageno
	case 3: Vdbe::Result_Text(fctx, cur->Pagetype, -1, DESTRUCTOR_STATIC); break; // pagetype
	case 4: Vdbe::Result_Int(fctx, cur->Cells); break; // ncell
	case 5: Vdbe::Result_Int(fctx, cur->Payload); break; // payload
	case 6: Vdbe::Result_Int(fctx, cur->Unused); break; // unused
	case 7: Vdbe::Result_Int(fctx, cur->MaxPayload); break; // mx_payload
	case 8: Vdbe::Result_Int64(fctx, cur->Offset); break; // pgoffset
	case 9: Vdbe::Result_Int(fctx, cur->SizePage); break; // pgsize
	}
	return RC_OK;
}

__device__ static RC statRowid(IVTableCursor *cursor, int64 *rowid)
{
	StatCursor *cur = (StatCursor *)cursor;
	*rowid = cur->Pageno;
	return RC_OK;
}

__constant__ static ITableModule _dbstat_module = {
	0,							// Version
	statConnect,				// Create
	statConnect,				// Connect
	statBestIndex,				// BestIndex
	statDisconnect,				// Disconnect
	statDisconnect,				// Destroy
	statOpen,					// Open - open a cursor
	statClose,					// Close - close a cursor
	statFilter,					// Filter - configure scan constraints
	statNext,					// Next - advance a cursor
	statEof,					// Eof - check for end of scan
	statColumn,					// Column - read data
	statRowid,					// Rowid - read data
	nullptr,                    // Update
	nullptr,                    // Begin
	nullptr,                    // Sync
	nullptr,                    // Commit
	nullptr,                    // Rollback
	nullptr,                    // FindMethod
	nullptr,                    // Rename
};

__device__ int sqlite3_dbstat_register(Context *ctx)
{
	VTable::CreateModule(ctx, "dbstat", &_dbstat_module, nullptr, nullptr);
	return RC_OK;
}

#endif

#if defined(_TEST) || TCLSH == 2
#include <JimEx.h>

static int test_dbstat(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
#ifdef OMIT_VIRTUALTABLE
	Jim_AppendResult(interp, "dbstat not available because of SQLITE_OMIT_VIRTUALTABLE", nullptr);
	return JIM_ERROR;
#else
	struct SqliteDb { Context *Ctx; };
	if (argc != 2)
	{
		Jim_WrongNumArgs(interp, 1, args, "DB");
		return JIM_ERROR;
	}
	Jim_CmdInfo cmdInfo;
	if (Jim_GetCommandInfo(interp, args[1], &cmdInfo))
	{
		Context *ctx = ((struct SqliteDb *)cmdInfo.objClientData)->Ctx;
		sqlite3_dbstat_register(ctx);
	}
	return JIM_OK;
#endif
}

__device__ int SqlitetestStat_Init(Jim_Interp *interp)
{
	Jim_CreateCommand(interp, "register_dbstat_vtab", test_dbstat, nullptr, nullptr);
	return JIM_OK;
}

#endif
