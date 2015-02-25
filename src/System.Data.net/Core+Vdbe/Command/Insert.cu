// insert.c
#include "..\VdbeInt.cu.h"

namespace Core { namespace Command
{
	__device__ void Insert::OpenTable(Parse *p, int cur, int db, Table *table, OP opcode)
	{
		_assert(!IsVirtual(table));
		Vdbe *v = p->GetVdbe();
		_assert(opcode == OP_OpenWrite || opcode == OP_OpenRead);
		p->TableLock(db, table->Id, (opcode == OP_OpenWrite), table->Name);
		v->AddOp3(opcode, cur, table->Id, db);
		v->ChangeP4(-1, (char *)INT_TO_PTR(table->Cols.length), Vdbe::P4T_INT32);
		v->Comment("%s", table->Name);
	}

	__device__ const char *Insert::IndexAffinityStr(Vdbe *v, Index *index)
	{
		if (!index->ColAff)
		{
			// The first time a column affinity string for a particular index is required, it is allocated and populated here. It is then stored as
			// a member of the Index structure for subsequent use.
			//
			// The column affinity string will eventually be deleted by sqliteDeleteIndex() when the Index structure itself is cleaned up.
			Table *table = index->Table;
			Context *ctx = v->Ctx;
			index->ColAff = (char *)_tagalloc(nullptr, index->Columns.length+2);
			if (!index->ColAff)
			{
				ctx->MallocFailed = true;
				return nullptr;
			}
			int n;
			for (n = 0; n < index->Columns.length; n++)
				index->ColAff[n] = table->Cols[index->Columns[n]].Affinity;
			index->ColAff[n++] = AFF_INTEGER;
			index->ColAff[n] = 0;
		}
		return index->ColAff;
	}

	__device__ void Insert::TableAffinityStr(Vdbe *v, Table *table)
	{
		// The first time a column affinity string for a particular table is required, it is allocated and populated here. It is then 
		// stored as a member of the Table structure for subsequent use.
		//
		// The column affinity string will eventually be deleted by sqlite3DeleteTable() when the Table structure itself is cleaned up.
		if (!table->ColAff)
		{
			Context *ctx = v->Ctx;
			char *colAff = (char *)_tagalloc(0, table->Cols.length+1);
			if (!colAff)
			{
				ctx->MallocFailed = true;
				return;
			}
			for (int i = 0; i < table->Cols.length; i++)
				colAff[i] = table->Cols[i].Affinity;
			colAff[table->Cols.length] = '\0';
			table->ColAff = colAff;
		}
		v->ChangeP4(-1, table->ColAff, Vdbe::P4T_TRANSIENT);
	}

	__device__ static bool ReadsTable(Parse *p, int startAddr, int db, Table *table)
	{
		Vdbe *v = p->GetVdbe();
		int end = v->CurrentAddr();
#ifndef OMIT_VIRTUALTABLE
		VTable *vtable = (IsVirtual(table) ? VTable::GetVTable(p->Ctx, table) : nullptr);
#endif
		for (int i = startAddr; i < end; i++)
		{
			Vdbe::VdbeOp *op = v->GetOp(i);
			_assert(op != nullptr);
			if (op->Opcode == OP_OpenRead && op->P3 == db)
			{
				int id = op->P2;
				if (id == table->Id)
					return true;
				for (Index *index = table->Index; index; index = index->Next)
					if (id == index->Id)
						return true;
			}
#ifndef OMIT_VIRTUALTABLE
			if (op->Opcode == OP_VOpen && op->P4.VTable == vtable)
			{
				_assert(op->P4.VTable != nullptr);
				_assert(op->P4Type == Vdbe::P4T_VTAB);
				return true;
			}
#endif
		}
		return false;
	}

#ifndef OMIT_AUTOINCREMENT
	__device__ static int AutoIncBegin(Parse *parse, int db, Table *table)
	{
		int memId = 0; // Register holding maximum rowid
		if (table->TabFlags & TF_Autoincrement)
		{
			Parse *toplevel = Parse_Toplevel(parse->Toplevel);
			AutoincInfo *info = toplevel->Ainc;
			while (info && info->Table  != table) info = info->Next;
			if (!info)
			{
				info = (AutoincInfo *)_tagalloc(parse->Ctx, sizeof(*info));
				if (!info) return 0;
				info->Next = toplevel->Ainc;
				toplevel->Ainc = info;
				info->Table = table;
				info->DB = db;
				toplevel->Mems++;					// Register to hold name of table
				info->RegCtr = ++toplevel->Mems;	// Max rowid register
				toplevel->Mems++;					// Rowid in sqlite_sequence
			}
			memId = info->RegCtr;
		}
		return memId;
	}

	__device__ void Insert::AutoincrementBegin(Parse *parse)
	{
		Context *ctx = parse->Ctx; // The database connection
		Vdbe *v = parse->V; // VDBE under construction

		// This routine is never called during trigger-generation.  It is only called from the top-level
		_assert(parse->TriggerTab == nullptr);
		_assert(parse == Parse_Toplevel(parse));

		_assert(v); // We failed long ago if this is not so
		for (AutoincInfo *p = parse->Ainc; p; p = p->Next) // Information about an AUTOINCREMENT
		{
			Context::DB *dbAsObj = &ctx->DBs[p->DB];  // Database only autoinc table
			int memId = p->RegCtr; // Register holding max rowid
			_assert(Btree::SchemaMutexHeld(ctx, 0, dbAsObj->Schema));
			OpenTable(parse, 0, p->DB, dbAsObj->Schema->SeqTable, OP_OpenRead);
			v->AddOp3(OP_Null, 0, memId, memId+1);
			int addr = v->CurrentAddr(); // A VDBE address
			v->AddOp4(OP_String8, 0, memId-1, 0, p->Table->Name, 0);
			v->AddOp2(OP_Rewind, 0, addr+9);
			v->AddOp3(OP_Column, 0, 0, memId);
			v->AddOp3(OP_Ne, memId-1, addr+7, memId);
			v->ChangeP5(AFF_BIT_JUMPIFNULL);
			v->AddOp2(OP_Rowid, 0, memId+1);
			v->AddOp3(OP_Column, 0, 1, memId);
			v->AddOp2(OP_Goto, 0, addr+9);
			v->AddOp2(OP_Next, 0, addr+2);
			v->AddOp2(OP_Integer, 0, memId);
			v->AddOp0(OP_Close);
		}
	}

	__device__ static void AutoIncStep(Parse *parse, int memId, int regRowid)
	{
		if (memId > 0)
			parse->V->AddOp2(OP_MemMax, memId, regRowid);
	}

	__device__ void Insert::AutoincrementEnd(Parse *parse)
	{
		Vdbe *v = parse->V;
		Context *ctx = parse->Ctx;

		_assert(v);
		for (AutoincInfo *p = parse->Ainc; p; p = p->Next)
		{
			Context::DB *dbAsObj = &ctx->DBs[p->DB];
			int memId = p->RegCtr;

			int rec = Expr::GetTempReg(parse);
			_assert(Btree::SchemaMutexHeld(ctx, 0, dbAsObj->Schema));
			OpenTable(parse, 0, p->DB, dbAsObj->Schema->SeqTable, OP_OpenWrite);
			int j1 = v->AddOp1(OP_NotNull, memId+1);
			int j2 = v->AddOp0(OP_Rewind);
			int j3 = v->AddOp3(OP_Column, 0, 0, rec);
			int j4 = v->AddOp3(OP_Eq, memId-1, 0, rec);
			v->AddOp2(OP_Next, 0, j3);
			v->JumpHere(j2);
			v->AddOp2(OP_NewRowid, 0, memId+1);
			int j5 = v->AddOp0(OP_Goto);
			v->JumpHere(j4);
			v->AddOp2(OP_Rowid, 0, memId+1);
			v->JumpHere(j1);
			v->JumpHere(j5);
			v->AddOp3(OP_MakeRecord, memId-1, 2, rec);
			v->AddOp3(OP_Insert, 0, rec, memId+1);
			v->ChangeP5(Vdbe::OPFLAG_APPEND);
			v->AddOp0(OP_Close);
			Expr::ReleaseTempReg(parse, rec);
		}
	}
#else
#define AutoIncBegin(A,B,C) (0)
#define AutoIncStep(A,B,C)
#endif

	__device__ RC Insert::CodeCoroutine(Parse *parse, Select *select, SelectDest *dest)
	{
		int regYield = ++parse->Mems; // Register holding co-routine entry-point
		int regEof = ++parse->Mems; // Register holding co-routine completion flag
		Vdbe *v = parse->GetVdbe(); // VDBE under construction
		int addrTop = v->CurrentAddr();					// Top of the co-routine
		v->AddOp2(OP_Integer, addrTop+2, regYield);		// X <- A
		v->Comment("Co-routine entry point");
		v->AddOp2(OP_Integer, 0, regEof);				// EOF <- 0
		v->Comment("Co-routine completion flag");
		Select::DestInit(dest, SRT_Coroutine, regYield);
		int j1 = v->AddOp2(OP_Goto, 0, 0);				// Jump instruction
		RC rc = Select::Select_(parse, select, dest);
		_assert(parse->Errs == 0 || rc);
		if (parse->Ctx->MallocFailed && rc == RC_OK) rc = RC_NOMEM;
		if (rc) return rc;
		v->AddOp2(OP_Integer, 1, regEof);				// EOF <- 1
		v->AddOp1(OP_Yield, regYield);					// yield X
		v->AddOp2(OP_Halt, RC_INTERNAL, OE_Abort);
		v->Comment("End of coroutine");
		v->JumpHere(j1);								// label B:
		return rc;
	}

	// Forward declaration
	__device__ static bool XferOptimization(Parse *parse, Table *dest, Select *select, OE onError, int dbDestId);
	__device__ void Insert::Insert_(Parse *parse, SrcList *tabList, ExprList *list, Select *select, IdList *column, OE onError)
	{
		int i, j, idx;        // Loop counters
		int baseCur = 0;      // VDBE Cursor number for pTab
		int keyColumn = -1;   // Column that is the INTEGER PRIMARY KEY

		Context *ctx = parse->Ctx; // The main database structure
		SelectDest dest; // Destination for SELECT on rhs of INSERT
		_memset(&dest, 0, sizeof(dest));
		if (parse->Errs || ctx->MallocFailed)
			goto insert_cleanup;

		// Locate the table into which we will be inserting new information.
		_assert(tabList->Srcs == 1);
		char *tableName = tabList->Ids[0].Name; // Name of the table into which we are inserting
		if (_NEVER(tableName==0) ) goto insert_cleanup;
		Table *table = Delete::SrcListLookup(parse, tabList); // The table to insert into.  aka TABLE
		if (!table)
			goto insert_cleanup;
		int db = Prepare::SchemaToIndex(ctx, table->Schema); // Index of database holding TABLE
		_assert( db<ctx->DBs.length);
#if !OMIT_AUTHORIZATION
		Context::DB *dbAsObj = &ctx->DBs[db]; // The database containing table being inserted into
		const char *dbName = dbAsObj->Name; // Name of the database holding this table
		if (Auth::Check(parse, AUTH_INSERT, table->Name, nullptr, dbName))
			goto insert_cleanup;
#endif

#ifndef OMIT_TRIGGER
		// Figure out if we have any triggers and if the table being inserted into is a view
		TRIGGER tmask; // Mask of trigger times
		Trigger *trigger = Trigger::TriggersExist(parse, table, TK_INSERT, 0, &tmask); // List of triggers on table, if required
		_assert((trigger && tmask) || (trigger == 0 && tmask == 0));
#ifndef OMIT_VIEW
		bool isView = (table->Select != nullptr); // True if attempting to insert into a view
#else
#define isView false
#endif
#else
#define trigger 0
#define tmask 0
#define isView 0
#endif

		// If table is really a view, make sure it has been initialized. ViewGetColumnNames() is a no-op if table is not a view (or virtual module table).
		if (parse->ViewGetColumnNames(table))
			goto insert_cleanup;

		// Ensure that:
		// (a) the table is not read-only, 
		// (b) that if it is a view then ON INSERT triggers exist
		if (Delete::IsReadOnly(parse, table, tmask))
			goto insert_cleanup;

		// Allocate a VDBE
		Vdbe *v = parse->GetVdbe(); // Generate code into this virtual machine
		if (!v) goto insert_cleanup;
		if (parse->Nested == 0) v->CountChanges();
		parse->BeginWriteOperation(select || trigger, db);

#ifndef OMIT_XFER_OPT
		// If the statement is of the form
		//
		//       INSERT INTO <table1> SELECT * FROM <table2>;
		//
		// Then special optimizations can be applied that make the transfer very fast and which reduce fragmentation of indices.
		//
		// This is the 2nd template.
		if (!column && XferOptimization(parse, table, select, onError, db))
		{
			_assert(!trigger);
			_assert(!list);
			goto insert_end;
		}
#endif

		// Register allocations
		int regFromSelect = 0;// Base register for data coming from SELECT
		int regAutoinc = 0;   // Register holding the AUTOINCREMENT counter
		int regRowCount = 0;  // Memory cell used for the row counter
		int regIns;           // Block of regs holding rowid+data being inserted
		int regRowid;         // registers holding insert rowid
		int regData;          // register holding first column to insert
		int regEof = 0;       // Register recording end of SELECT data
		int *regIdxs = nullptr;     // One register allocated to each index

		// If this is an AUTOINCREMENT table, look up the sequence number in the sqlite_sequence table and store it in memory cell regAutoinc.
		regAutoinc = AutoIncBegin(parse, db, table);

		// Figure out how many columns of data are supplied.  If the data is coming from a SELECT statement, then generate a co-routine that
		// produces a single row of the SELECT on each invocation.  The co-routine is the common header to the 3rd and 4th templates.
		int columns; // Number of columns in the data
		bool useTempTable = false; // Store SELECT results in intermediate table
		int srcTab = 0; // Data comes from this temporary cursor if >=0
		int addrSelect = 0; // Address of coroutine that implements the SELECT
		if (select)
		{
			// Data is coming from a SELECT.  Generate a co-routine to run that SELECT.
			RC rc = CodeCoroutine(parse, select, &dest);
			if (rc) goto insert_cleanup;

			regEof = dest.SDParmId + 1;
			regFromSelect = dest.SdstId;
			_assert(select->EList);
			columns = select->EList->Exprs;
			_assert(dest.SdstId == columns);

			// Set useTempTable to TRUE if the result of the SELECT statement should be written into a temporary table (template 4).  Set to
			// FALSE if each* row of the SELECT can be written directly into the destination table (template 3).
			//
			// A temp table must be used if the table being updated is also one of the tables being read by the SELECT statement.  Also use a 
			// temp table in the case of row triggers.
			if (trigger || ReadsTable(parse, addrSelect, db, table))
				useTempTable = true;

			if (useTempTable)
			{
				// Invoke the coroutine to extract information from the SELECT and add it to a transient table srcTab.  The code generated
				// here is from the 4th template:
				//
				//      B: open temp table
				//      L: yield X
				//         if EOF goto M
				//         insert row from R..R+n into temp table
				//         goto L
				//      M: ...
				srcTab = parse->Tabs++;
				int regRec = Expr::GetTempReg(parse);				// Register to hold packed record
				int regTempRowid = Expr::GetTempReg(parse);			// Register to hold temp table ROWID
				v->AddOp2(OP_OpenEphemeral, srcTab, columns);
				int addrTop = v->AddOp1(OP_Yield, dest.SDParmId);	// Label "L"
				int addrIf = v->AddOp1(OP_If, regEof);				// Address of jump to M
				v->AddOp3(OP_MakeRecord, regFromSelect, columns, regRec);
				v->AddOp2(OP_NewRowid, srcTab, regTempRowid);
				v->AddOp3(OP_Insert, srcTab, regRec, regTempRowid);
				v->AddOp2(OP_Goto, 0, addrTop);
				v->JumpHere(addrIf);
				Expr::ReleaseTempReg(parse, regRec);
				Expr::ReleaseTempReg(parse, regTempRowid);
			}
		}
		else
		{
			// This is the case if the data for the INSERT is coming from a VALUES clause
			NameContext sNC;
			_memset(&sNC, 0, sizeof(sNC));
			sNC.Parse = parse;
			srcTab = -1;
			_assert(!useTempTable);
			columns = (list ? list->Exprs : 0);
			for (i = 0; i < columns; i++)
				if (Walker::ResolveExprNames(&sNC, list->Ids[i].Expr))
					goto insert_cleanup;
		}

		// Make sure the number of columns in the source data matches the number of columns to be inserted into the table.
		int hidden = 0; // Number of hidden columns if TABLE is virtual
		if (IsVirtual(table))
			for (i = 0; i < table->Cols.length; i++)
				hidden += (IsHiddenColumn(&table->Cols[i]) ? 1 : 0);
		if (!column && columns && columns != (table->Cols.length - hidden))
		{
			parse->ErrorMsg("table %S has %d columns but %d values were supplied", tabList, 0, table->Cols.length - hidden, columns);
			goto insert_cleanup;
		}
		if (column != nullptr && columns != column->Ids.length)
		{
			parse->ErrorMsg("%d values for %d columns", columns, column->Ids.length);
			goto insert_cleanup;
		}

		// If the INSERT statement included an IDLIST term, then make sure all elements of the IDLIST really are columns of the table and 
		// remember the column indices.
		//
		// If the table has an INTEGER PRIMARY KEY column and that column is named in the IDLIST, then record in the keyColumn variable
		// the index into IDLIST of the primary key column.  keyColumn is the index of the primary key as it appears in IDLIST, not as
		// is appears in the original table.  (The index of the primary key in the original table is table->iPKey.)
		if (column)
		{
			for (i = 0; i < column->Ids.length; i++)
				column->Ids[i].Idx = -1;
			for (i = 0; i < column->Ids.length; i++)
			{
				for (j = 0; j < table->Cols.length; j++)
				{
					if (!_strcmp(column->Ids[i].Name, table->Cols[j].Name))
					{
						column->Ids[i].Idx = j;
						if (j == table->PKey)
							keyColumn = i;
						break;
					}
				}
				if (j >= table->Cols.length)
				{
					if (Expr::IsRowid(column->Ids[i].Name))
						keyColumn = i;
					else
					{
						parse->ErrorMsg("table %S has no column named %s", tabList, 0, column->Ids[i].Name);
						parse->CheckSchema = 1;
						goto insert_cleanup;
					}
				}
			}
		}

		// If there is no IDLIST term but the table has an integer primary key, the set the keyColumn variable to the primary key column index
		// in the original table definition.
		if (!column && columns > 0)
			keyColumn = table->PKey;

		// Initialize the count of rows to be inserted
		if (ctx->Flags & Context::FLAG_CountRows)
		{
			regRowCount = ++parse->Mems;
			v->AddOp2(OP_Integer, 0, regRowCount);
		}

		// If this is not a view, open the table and and all indices
		if (!isView)
		{
			baseCur = parse->Tabs;
			int idxs = OpenTableAndIndices(parse, table, baseCur, OP_OpenWrite);
			regIdxs = (int *)_tagalloc(ctx, sizeof(int)*(idxs+1));
			if (!regIdxs)
				goto insert_cleanup;
			for (i = 0; i < idxs; i++)
				regIdxs[i] = ++parse->Mems;
		}

		// This is the top of the main insertion loop
		int addrInsTop = 0; // Jump to label "D"
		int addrCont = 0; // Top of insert loop. Label "C" in templates 3 and 4
		if (useTempTable)
		{
			// This block codes the top of loop only.  The complete loop is the following pseudocode (template 4):
			//
			//         rewind temp table
			//      C: loop over rows of intermediate table
			//           transfer values form intermediate table into <table>
			//         end loop
			//      D: ...
			addrInsTop = v->AddOp1(OP_Rewind, srcTab);
			addrCont = v->CurrentAddr();
		}
		else if (select)
		{
			// This block codes the top of loop only.  The complete loop is the following pseudocode (template 3):
			//
			//      C: yield X
			//         if EOF goto D
			//         insert the select result into <table> from R..R+n
			//         goto C
			//      D: ...
			addrCont = v->AddOp1(OP_Yield, dest.SDParmId);
			addrInsTop = v->AddOp1(OP_If, regEof);
		}

		// Allocate registers for holding the rowid of the new row, the content of the new row, and the assemblied row record.
		regRowid = regIns = parse->Mems+1;
		parse->Mems += table->Cols.length + 1;
		if (IsVirtual(table))
		{
			regRowid++;
			parse->Mems++;
		}
		regData = regRowid+1;

		// Run the BEFORE and INSTEAD OF triggers, if there are any
		int endOfLoop = v->MakeLabel(); // Label for the end of the insertion loop
		if (tmask & TRIGGER_BEFORE)
		{
			int regCols = Expr::GetTempRange(parse, table->Cols.length+1);

			// build the NEW.* reference row.  Note that if there is an INTEGER PRIMARY KEY into which a NULL is being inserted, that NULL will be
			// translated into a unique ID for the row.  But on a BEFORE trigger, we do not know what the unique ID will be (because the insert has
			// not happened yet) so we substitute a rowid of -1
			if (keyColumn < 0)
				v->AddOp2(OP_Integer, -1, regCols);
			else
			{
				if (useTempTable)
					v->AddOp3(OP_Column, srcTab, keyColumn, regCols);
				else
				{
					_assert(!select); // Otherwise useTempTable is true
					Expr::Code(parse, list->Ids[keyColumn].Expr, regCols);
				}
				int j1 = v->AddOp1(OP_NotNull, regCols);
				v->AddOp2(OP_Integer, -1, regCols);
				v->JumpHere(j1);
				v->AddOp1(OP_MustBeInt, regCols);
			}

			// Cannot have triggers on a virtual table. If it were possible, this block would have to account for hidden column.
			_assert(!IsVirtual(table));

			// Create the new column data
			for (i = 0; i < table->Cols.length; i++)
			{
				if (!column)
					j = i;
				else
					for (j = 0; j < column->Ids.length; j++)
						if (column->Ids[j].Idx == i) break;
				if ((!useTempTable && !list) || (column && j >= column->Ids.length))
					Expr::Code(parse, table->Cols[i].Dflt, regCols+i+1);
				else if (useTempTable)
					v->AddOp3(OP_Column, srcTab, j, regCols+i+1); 
				else
				{
					_assert(!select); // Otherwise useTempTable is true
					Expr::CodeAndCache(parse, list->Ids[j].Expr, regCols+i+1);
				}
			}

			// If this is an INSERT on a view with an INSTEAD OF INSERT trigger, do not attempt any conversions before assembling the record.
			// If this is a real table, attempt conversions as required by the table column affinities.
			if (!isView)
			{
				v->AddOp2(OP_Affinity, regCols+1, table->Cols.length);
				TableAffinityStr(v, table);
			}

			// Fire BEFORE or INSTEAD OF triggers
			trigger->CodeRowTrigger(parse, TK_INSERT, nullptr, TRIGGER_BEFORE, table, regCols - table->Cols.length-1, onError, endOfLoop);

			Expr::ReleaseTempRange(parse, regCols, table->Cols.length+1);
		}

		// Push the record number for the new entry onto the stack.  The record number is a randomly generate integer created by NewRowid
		// except when the table has an INTEGER PRIMARY KEY column, in which case the record number is the same as that column.
		bool appendFlag = false; // True if the insert is likely to be an append
		if (!isView)
		{
			if (IsVirtual(table))
				v->AddOp2(OP_Null, 0, regIns); // The row that the VUpdate opcode will delete: none
			if (keyColumn >= 0)
			{
				if (useTempTable)
					v->AddOp3(OP_Column, srcTab, keyColumn, regRowid);
				else if (select)
					v->AddOp2(OP_SCopy, regFromSelect+keyColumn, regRowid);
				else
				{
					Expr::Code(parse, list->Ids[keyColumn].Expr, regRowid);
					Vdbe::VdbeOp *op = v->GetOp(-1);
					if (_ALWAYS(op) && op->Opcode == OP_Null && !IsVirtual(table))
					{
						appendFlag = true;
						op->Opcode = OP_NewRowid;
						op->P1 = baseCur;
						op->P2 = regRowid;
						op->P3 = regAutoinc;
					}
				}
				// If the PRIMARY KEY expression is NULL, then use OP_NewRowid to generate a unique primary key value.
				if (!appendFlag)
				{
					int j1;
					if (!IsVirtual(table))
					{
						j1 = v->AddOp1(OP_NotNull, regRowid);
						v->AddOp3(OP_NewRowid, baseCur, regRowid, regAutoinc);
						v->JumpHere(j1);
					}
					else
					{
						j1 = v->CurrentAddr();
						v->AddOp2(OP_IsNull, regRowid, j1+2);
					}
					v->AddOp1(OP_MustBeInt, regRowid);
				}
			}
			else if (IsVirtual(table))
				v->AddOp2(OP_Null, 0, regRowid);
			else
			{
				v->AddOp3(OP_NewRowid, baseCur, regRowid, regAutoinc);
				appendFlag = true;
			}
			AutoIncStep(parse, regAutoinc, regRowid);

			// Push onto the stack, data for all columns of the new entry, beginning with the first column.
			hidden = 0;
			for (i = 0; i < table->Cols.length; i++)
			{
				int regStoreId = regRowid+1+i;
				if (i == table->PKey)
				{
					// The value of the INTEGER PRIMARY KEY column is always a NULL. Whenever this column is read, the record number will be substituted
					// in its place.  So will fill this column with a NULL to avoid taking up data space with information that will never be used.
					v->AddOp2(OP_Null, 0, regStoreId);
					continue;
				}
				if (!column)
				{
					if (IsHiddenColumn(&table->Cols[i]))
					{
						_assert(IsVirtual(table));
						j = -1;
						hidden++;
					}
					else
						j = i - hidden;
				}
				else
					for (j = 0; j < column->Ids.length; j++)
						if (column->Ids[j].Idx == i) break;
				if (j < 0 || columns == 0 || (column && j >= column->Ids.length))
					Expr::Code(parse, table->Cols[i].Dflt, regStoreId);
				else if (useTempTable)
					v->AddOp3(OP_Column, srcTab, j, regStoreId); 
				else if (select)
					v->AddOp2(OP_SCopy, regFromSelect+j, regStoreId);
				else
					Expr::Code(parse, list->Ids[j].Expr, regStoreId);
			}

			// Generate code to check constraints and generate index keys and do the insertion.
#ifndef OMIT_VIRTUALTABLE
			if (IsVirtual(table))
			{
				const char *vtable = (const char *)VTable::GetVTable(ctx, table);
				VTable::MakeWritable(parse, table);
				v->AddOp4(OP_VUpdate, 1, table->Cols.length+2, regIns, vtable, Vdbe::P4T_VTAB);
				v->ChangeP5(onError == OE_Default ? OE_Abort : onError);
				parse->MayAbort();
			}
			else
#endif
			{
				bool isReplace; // Set to true if constraints may cause a replace
				GenerateConstraintChecks(parse, table, baseCur, regIns, regIdxs, keyColumn >= 0, 0, onError, endOfLoop, &isReplace);
				parse->FKCheck(table, 0, regIns);
				CompleteInsertion(parse, table, baseCur, regIns, regIdxs, 0, appendFlag, !isReplace);
			}
		}

		// Update the count of rows that are inserted
		if ((ctx->Flags & Context::FLAG_CountRows) != 0)
			v->AddOp2(OP_AddImm, regRowCount, 1);

		if (trigger) // Code AFTER triggers
			trigger->CodeRowTrigger(parse, TK_INSERT, 0, TRIGGER_AFTER, table, regData-2 - table->Cols.length, onError, endOfLoop);

		// The bottom of the main insertion loop, if the data source is a SELECT statement.
		v->ResolveLabel(endOfLoop);
		if (useTempTable)
		{
			v->AddOp2(OP_Next, srcTab, addrCont);
			v->JumpHere( addrInsTop);
			v->AddOp1(OP_Close, srcTab);
		}
		else if (select)
		{
			v->AddOp2(OP_Goto, 0, addrCont);
			v->JumpHere(addrInsTop);
		}

		if (!IsVirtual(table) && !isView)
		{
			// Close all tables opened
			v->AddOp1(OP_Close, baseCur);
			Index *index; // For looping over indices of the table
			for (idx=1, index = table->Index; index; index = index->Next, idx++)
				v->AddOp1(OP_Close, idx+baseCur);
		}

insert_end:
		// Update the sqlite_sequence table by storing the content of the maximum rowid counter values recorded while inserting into autoincrement tables.
		if (parse->Nested == 0 && parse->TriggerTab == 0)
			AutoincrementEnd(parse);

		// Return the number of rows inserted. If this routine is generating code because of a call to sqlite3NestedParse(), do not invoke the callback function.
		if ((ctx->Flags & Context::FLAG_CountRows) && !parse->Nested && !parse->TriggerTab)
		{
			v->AddOp2(OP_ResultRow, regRowCount, 1);
			v->SetNumCols(1);
			v->SetColName(0, COLNAME_NAME, "rows inserted", DESTRUCTOR_STATIC);
		}

insert_cleanup:
		Parse::SrcListDelete(ctx, tabList);
		Expr::ListDelete(ctx, list);
		Select::Delete(ctx, select);
		Parse::IdListDelete(ctx, column);
		_tagfree(ctx, regIdxs);
	}

	// Make sure "isView" and other macros defined above are undefined. Otherwise thely may interfere with compilation of other functions in this file
	// (or in another file, if this file becomes part of the amalgamation).
#ifdef isView
#undef isView
#endif
#ifdef pTrigger
#undef pTrigger
#endif
#ifdef tmask
#undef tmask
#endif

	__device__ void Insert::GenerateConstraintChecks(Parse *parse, Table *table, int baseCur, int regRowid, int *regIdxs, int rowidChng, int isUpdate, OE overrideError, int ignoreDest, bool *mayReplaceOut)
	{
		Context *ctx = parse->Ctx; // Database connection
		Vdbe *v = parse->GetVdbe(); // VDBE under constrution
		_assert(v != nullptr);
		_assert(!table->Select); // This table is not a VIEW
		int colsLength = table->Cols.length; // Number of columns
		int regData = regRowid + 1; // Register containing first data column
		OE onError;  // Conflict resolution strategy

		// Test all NOT NULL constraints.
		int i;
		for (i = 0; i < colsLength; i++)
		{
			if (i == table->PKey)
				continue;
			onError = table->Cols[i].NotNull;
			if (onError == OE_None) continue;
			if (overrideError != OE_Default) onError = overrideError;
			else if (onError == OE_Default) onError = OE_Abort;
			if (onError == OE_Replace && !table->Cols[i].Dflt) onError = OE_Abort;
			_assert(onError == OE_Rollback || onError == OE_Abort || onError == OE_Fail || onError == OE_Ignore || onError == OE_Replace);
			switch (onError)
			{
			case OE_Abort:
				parse->MayAbort(); // Fall thru into the next case
			case OE_Rollback:
			case OE_Fail: {
				v->AddOp3(OP_HaltIfNull, RC_CONSTRAINT_NOTNULL, onError, regData+i);
				char *msg = _mtagprintf(ctx, "%s.%s may not be NULL", table->Name, table->Cols[i].Name);
				v->ChangeP4(-1, msg, Vdbe::P4T_DYNAMIC);
				break; }
			case OE_Ignore: {
				v->AddOp2(OP_IsNull, regData+i, ignoreDest);
				break; }
			default: {
				_assert(onError == OE_Replace);
				int j1 = v->AddOp1(OP_NotNull, regData+i); // Addresss of jump instruction
				Expr::Code(parse, table->Cols[i].Dflt, regData+i);
				v->JumpHere(j1);
				break; }
			}
		}

#ifndef OMIT_CHECK
		// Test all CHECK constraints
		if (table->Check && (ctx->Flags & Context::FLAG_IgnoreChecks) == 0)
		{
			ExprList *check = table->Check;
			parse->CkBase = regData;
			onError = (overrideError != OE_Default ? overrideError : OE_Abort);
			for (i = 0; i < check->Exprs; i++)
			{
				int allOk = v->MakeLabel();
				check->Ids[i].Expr->IfTrue(parse, allOk, AFF_BIT_JUMPIFNULL);
				if (onError == OE_Ignore)
					v->AddOp2(OP_Goto, 0, ignoreDest);
				else
				{
					if (onError == OE_Replace) onError = OE_Abort; // IMP: R-15569-63625
					char *consName = check->Ids[i].Name;
					consName = (consName ? _mtagprintf(ctx, "constraint %s failed", consName) : nullptr);
					parse->HaltConstraint(RC_CONSTRAINT_CHECK, onError, consName, Vdbe::P4T_DYNAMIC);
				}
				v->ResolveLabel(allOk);
			}
		}
#endif

		// If we have an INTEGER PRIMARY KEY, make sure the primary key of the new record does not previously exist.  Except, if this
		// is an UPDATE and the primary key is not changing, that is OK.
		bool seenReplace = false; // True if REPLACE is used to resolve INT PK conflict
		int j2 = 0, j3; // Addresses of jump instructions
		if (rowidChng)
		{
			onError = table->KeyConf;
			if (overrideError != OE_Default) onError = overrideError;
			else if (onError == OE_Default) onError = OE_Abort;
			if (isUpdate)
				j2 = v->AddOp3(OP_Eq, regRowid, 0, rowidChng);
			j3 = v->AddOp3(OP_NotExists, baseCur, 0, regRowid);
			switch (onError)
			{
			default: {
				onError = OE_Abort; } // Fall thru into the next case
			case OE_Rollback:
			case OE_Abort:
			case OE_Fail: {
				parse->HaltConstraint(RC_CONSTRAINT_PRIMARYKEY, onError, "PRIMARY KEY must be unique", Vdbe::P4T_STATIC);
				break; }
			case OE_Replace: {
				// If there are DELETE triggers on this table and the recursive-triggers flag is set, call GenerateRowDelete() to
				// remove the conflicting row from the table. This will fire the triggers and remove both the table and index b-tree entries.
				//
				// Otherwise, if there are no triggers or the recursive-triggers flag is not set, but the table has one or more indexes, call 
				// GenerateRowIndexDelete(). This removes the index b-tree entries only. The table b-tree entry will be replaced by the new entry 
				// when it is inserted.  
				//
				// If either GenerateRowDelete() or GenerateRowIndexDelete() is called, also invoke MultiWrite() to indicate that this VDBE may require
				// statement rollback (if the statement is aborted after the delete takes place). Earlier versions called sqlite3MultiWrite() regardless,
				// but being more selective here allows statements like:
				//
				//   REPLACE INTO t(rowid) VALUES($newrowid)
				//
				// to run without a statement journal if there are no indexes on the table.
				Trigger *trigger = nullptr;
				if (ctx->Flags & Context::FLAG_RecTriggers)
					trigger = Trigger::TriggersExist(parse, table, TK_DELETE, nullptr, nullptr);
				if (trigger || parse->FKRequired(table, 0, 0))
				{
					parse->MultiWrite();
					Delete::GenerateRowDelete(parse, table, baseCur, regRowid, 0, trigger, OE_Replace);
				}
				else if (table->Index)
				{
					parse->MultiWrite();
					Delete::GenerateRowIndexDelete(parse, table, baseCur, 0);
				}
				seenReplace = true;
				break; }
			case OE_Ignore: {
				_assert(!seenReplace);
				v->AddOp2(OP_Goto, 0, ignoreDest);
				break; }
			}
			v->JumpHere(j3);
			if (isUpdate)
				v->JumpHere(j2);
		}

		// Test all UNIQUE constraints by creating entries for each UNIQUE index and making sure that duplicate entries do not already exist.
		// Add the new records to the indices as we go.
		int curId; // Table cursor number
		Index *index; // Pointer to one of the indices
		int regOldRowid = (rowidChng && isUpdate ? rowidChng : regRowid);
		for (curId = 0, index = table->Index; index; index = index->Next, curId++)
		{
			if (regIdxs[curId] == 0) continue; // Skip unused indices

			// Create a key for accessing the index entry
			int regIdx = Expr::GetTempRange(parse, index->Columns.length+1);
			for (i = 0; i < index->Columns.length; i++)
			{
				int idx = index->Columns[i];
				if (idx == table->PKey)
					v->AddOp2(OP_SCopy, regRowid, regIdx+i);
				else
					v->AddOp2(OP_SCopy, regData+idx, regIdx+i);
			}
			v->AddOp2(OP_SCopy, regRowid, regIdx+i);
			v->AddOp3(OP_MakeRecord, regIdx, index->Columns.length+1, regIdxs[curId]);
			v->ChangeP4(-1, IndexAffinityStr(v, index), Vdbe::P4T_TRANSIENT);
			Expr::CacheAffinityChange(parse, regIdx, index->Columns.length+1);

			// Find out what action to take in case there is an indexing conflict
			onError = index->OnError;
			if (onError == OE_None)
			{ 
				Expr::ReleaseTempRange(parse, regIdx, index->Columns.length+1);
				continue; // index is not a UNIQUE index
			}
			if (overrideError != OE_Default) onError = overrideError;
			else if (onError == OE_Default) onError = OE_Abort;
			if (seenReplace)
			{
				if (onError == OE_Ignore) onError = OE_Replace;
				else if (onError == OE_Fail) onError = OE_Abort;
			}

			// Check to see if the new index entry will be unique
			int regR = Expr::GetTempReg(parse);
			v->AddOp2(OP_SCopy, regOldRowid, regR);
			j3 = v->AddOp4(OP_IsUnique, baseCur+curId+1, 0, regR, (char *)INT_TO_PTR(regIdx), Vdbe::P4T_INT32);
			Expr::ReleaseTempRange(parse, regIdx, index->Columns.length+1);

			// Generate code that executes if the new index entry is not unique
			_assert(onError == OE_Rollback || onError == OE_Abort || onError == OE_Fail || onError == OE_Ignore || onError == OE_Replace);
			switch (onError)
			{
			case OE_Rollback:
			case OE_Abort:
			case OE_Fail: {
				TextBuilder b;
				TextBuilder::Init(&b, nullptr, 0, 200);
				b.Tag = ctx;
				const char *sepText = (index->Columns.length > 1 ? "columns " : "column ");
				for (int j = 0; j < index->Columns.length; j++)
				{
					char *colName = table->Cols[index->Columns[j]].Name;
					b.Append(sepText, -1);
					sepText = ", ";
					b.Append(colName, -1);
				}
				b.Append((index->Columns.length > 1 ? " are not unique" : " is not unique"), -1);
				char *err = b.ToString();
				parse->HaltConstraint(RC_CONSTRAINT_UNIQUE, onError, err, 0);
				_tagfree(ctx, err);
				break; }
			case OE_Ignore: {
				_assert(!seenReplace);
				v->AddOp2(OP_Goto, 0, ignoreDest);
				break; }
			default: {
				Trigger *trigger = nullptr;
				_assert(onError == OE_Replace);
				parse->MultiWrite();
				if (ctx->Flags & Context::FLAG_RecTriggers)
					trigger = Trigger::TriggersExist(parse, table, TK_DELETE, 0, 0);
				Delete::GenerateRowDelete(parse, table, baseCur, regR, 0, trigger, OE_Replace);
				seenReplace = true;
				break; }
			}
			v->JumpHere(j3);
			Expr::ReleaseTempReg(parse, regR);
		}
		if (mayReplaceOut)
			*mayReplaceOut = seenReplace;
	}

	__device__ void Insert::CompleteInsertion(Parse *parse, Table *table, int baseCur, int regRowid, int *regIdxs, bool isUpdate, bool appendBias, bool useSeekResult)
	{
		Vdbe *v = parse->GetVdbe();
		_assert(v);
		_assert(!table->Select);  // This table is not a VIEW
		int indexsLength;
		Index *index;
		for (indexsLength = 0, index = table->Index; index; index = index->Next, indexsLength++) { }
		for (int i = indexsLength-1; i >= 0; i--)
		{
			if (regIdxs[i] == 0) continue;
			v->AddOp2(OP_IdxInsert, baseCur+i+1, regIdxs[i]);
			if (useSeekResult)
				v->ChangeP5(Vdbe::OPFLAG_USESEEKRESULT);
		}
		int regData = regRowid + 1;
		int regRec = Expr::GetTempReg(parse);
		v->AddOp3(OP_MakeRecord, regData, table->Cols.length, regRec);
		TableAffinityStr(v, table);
		Expr::CacheAffinityChange(parse, regData, table->Cols.length);
		Vdbe::OPFLAG pik_flags;
		if (parse->Nested)
			pik_flags = (Vdbe::OPFLAG)0;
		else
		{
			pik_flags = Vdbe::OPFLAG_NCHANGE;
			pik_flags |= (isUpdate ? Vdbe::OPFLAG_ISUPDATE : Vdbe::OPFLAG_LASTROWID);
		}
		if (appendBias) pik_flags |= Vdbe::OPFLAG_APPEND;
		if (useSeekResult) pik_flags |= Vdbe::OPFLAG_USESEEKRESULT;
		v->AddOp3(OP_Insert, baseCur, regRec, regRowid);
		if (!parse->Nested)
			v->ChangeP4(-1, table->Name, Vdbe::P4T_TRANSIENT);
		v->ChangeP5(pik_flags);
	}

	__device__ int Insert::OpenTableAndIndices(Parse *parse, Table *table, int baseCur, OP op)
	{
		if (IsVirtual(table)) return 0;
		int db = Prepare::SchemaToIndex(parse->Ctx, table->Schema);
		Vdbe *v = parse->GetVdbe();
		_assert(v);
		OpenTable(parse, baseCur, db, table, op);
		int i;
		Index *index;
		for (i = 1, index = table->Index; index; index = index->Next, i++)
		{
			KeyInfo *key = parse->IndexKeyinfo(index);
			_assert(index->Schema == table->Schema);
			v->AddOp4(op, i+baseCur, index->Id, db, (char *)key, Vdbe::P4T_KEYINFO_HANDOFF);
			v->Comment("%s", index->Name);
		}
		if (parse->Tabs < baseCur+i)
			parse->Tabs = baseCur+i;
		return i-1;
	}

#ifdef TEST
	__device__ int _xferopt_count;
#endif

#ifndef OMIT_XFER_OPT
	__device__ static bool XferCompatibleCollation(const char *z1, const char *z2)
	{
		if (!z1) return (z2 == nullptr);
		if (!z2) return false;
		return !_strcmp(z1, z2);
	}

	__device__ static bool XferCompatibleIndex(Index *dest, Index *src)
	{
		_assert(dest && src);
		_assert(dest->Table != src->Table);
		if (dest->Columns.length != src->Columns.length) return false;			// Different number of columns
		if (dest->OnError != src->OnError) return false;						// Different conflict resolution strategies
		for (int i = 0; i < src->Columns.length; i++)
		{
			if (src->Columns[i] != dest->Columns[i]) return false;				// Different columns indexed
			if (src->SortOrders[i] != dest->SortOrders[i]) return false;		// Different sort orders
			if (!XferCompatibleCollation(src->CollNames[i], dest->CollNames[i])) return false; // Different collating sequences
		}
		// If no test above fails then the indices must be compatible
		return true;
	}

	__device__ static bool XferOptimization(Parse *parse, Table *dest, Select *select, OE onError, int dbDestId)
	{
		if (!select) return false;									// Must be of the form  INSERT INTO ... SELECT ...
		if (Trigger::List(parse, dest)) return false;			// tab1 must not have triggers
#ifndef OMIT_VIRTUALTABLE
		if (dest->TabFlags & TF_Virtual) return false;				// tab1 must not be a virtual table
#endif
		if (onError == OE_Default)
		{
			if (dest->PKey >= 0) onError = dest->KeyConf;
			if (onError == OE_Default) onError = OE_Abort;
		}
		_assert(select->Src);										// allocated even if there is no FROM clause
		if (select->Src->Srcs != 1) return false;					// FROM clause must have exactly one term
		if (select->Src->Ids[0].Select) return false;				// FROM clause cannot contain a subquery
		if (select->Where) return false;							// SELECT may not have a WHERE clause
		if (select->OrderBy) return false;							// SELECT may not have an ORDER BY clause
		// Do not need to test for a HAVING clause.  If HAVING is present but there is no ORDER BY, we will get an error.
		if (select->GroupBy) return false;							// SELECT may not have a GROUP BY clause
		if (select->Limit) return false;							// SELECT may not have a LIMIT clause
		_assert(select->Offset == nullptr);								// Must be so if pLimit==0
		if (select->Prior) return false;							// SELECT may not be a compound query
		if (select->SelFlags & SF_Distinct) return false;			// SELECT may not be DISTINCT
		ExprList *list = select->EList;							// The result set of the SELECT
		_assert(list != nullptr);
		if (list->Exprs != 1) return false;						// The result set must have exactly one column
		_assert(list->Ids[0].Expr);
		if (list->Ids[0].Expr->OP != TK_ALL) return false;			// The result set must be the special operator "*"

		// At this point we have established that the statement is of the correct syntactic form to participate in this optimization.  Now
		// we have to check the semantics.
		SrcList::SrcListItem *item = select->Src->Ids;				// An element of pSelect->pSrc
		Table *src = parse->LocateTableItem(false, item);			// The table in the FROM clause of SELECT
		if (!src) return false;										// FROM clause does not contain a real table
		if (src == dest) return false;								// tab1 and tab2 may not be the same table
#ifndef OMIT_VIRTUALTABLE
		if (src->TabFlags & TF_Virtual) return false;				// tab2 must not be a virtual table
#endif
		if (src->Select) return false;								// tab2 may not be a view
		if (dest->Cols.length != src->Cols.length) return false;	// Number of columns must be the same in tab1 and tab2
		if (dest->PKey != src->PKey) return false;					// Both tables must have the same INTEGER PRIMARY KEY
		for (int i = 0; i < dest->Cols.length; i++)
		{
			if (dest->Cols[i].Affinity != src->Cols[i].Affinity) return false; // Affinity must be the same on all columns
			if (!XferCompatibleCollation(dest->Cols[i].Coll, src->Cols[i].Coll)) return false; // Collating sequence must be the same on all columns
			if (dest->Cols[i].NotNull && !src->Cols[i].NotNull) return false; // tab2 must be NOT NULL if tab1 is
		}
		bool destHasUniqueIdx = false; // True if pDest has a UNIQUE index
		Index *srcIdx, *destIdx; // Source and destination indices
		for (destIdx = dest->Index; destIdx; destIdx = destIdx->Next)
		{
			if (destIdx->OnError != OE_None) destHasUniqueIdx = true;
			for (srcIdx = src->Index; srcIdx; srcIdx = srcIdx->Next)
				if (XferCompatibleIndex(destIdx, srcIdx)) break;
			if (!srcIdx) return false; // pDestIdx has no corresponding index in pSrc
		}
#ifndef OMIT_CHECK
		if (dest->Check && Expr::ListCompare(src->Check, dest->Check)) return false; // Tables have different CHECK constraints.  Ticket #2252
#endif
#ifndef OMIT_FOREIGN_KEY
		// Disallow the transfer optimization if the destination table constains any foreign key constraints.  This is more restrictive than necessary.
		// But the main beneficiary of the transfer optimization is the VACUUM command, and the VACUUM command disables foreign key constraints.  So
		// the extra complication to make this rule less restrictive is probably not worth the effort.  Ticket [6284df89debdfa61db8073e062908af0c9b6118e]
		if ((parse->Ctx->Flags & Context::FLAG_ForeignKeys) != 0 && dest->FKeys != nullptr) return false;
#endif
		if ((parse->Ctx->Flags & Context::FLAG_CountRows) != 0) return false;  // xfer opt does not play well with PRAGMA count_changes

		// If we get this far, it means that the xfer optimization is at least a possibility, though it might only work if the destination
		// table (tab1) is initially empty.
#ifdef TEST
		_xferopt_count++;
#endif
		int dbSrcId = Prepare::SchemaToIndex(parse->Ctx, src->Schema); // The database of pSrc
		Vdbe *v = parse->GetVdbe(); // The VDBE we are building
		parse->CodeVerifySchema(dbSrcId);
		int srcId = parse->Tabs++; // Cursors from source and destination
		int destId = parse->Tabs++; // Cursors from source and destination
		int regAutoinc = AutoIncBegin(parse, dbDestId, dest); // Memory register used by AUTOINC
		Insert::OpenTable(parse, destId, dbDestId, dest, OP_OpenWrite);
		int addr1; // Loop addresses
		int emptyDestTest; // Address of test for empty pDest
		if ((dest->PKey < 0 && dest->Index != nullptr) ||		// (1)
			destHasUniqueIdx ||									// (2)
			(onError != OE_Abort && onError != OE_Rollback))	// (3)
		{
			// In some circumstances, we are able to run the xfer optimization only if the destination table is initially empty.  This code makes
			// that determination.  Conditions under which the destination must be empty:
			//
			// (1) There is no INTEGER PRIMARY KEY but there are indices. (If the destination is not initially empty, the rowid fields
			//     of index entries might need to change.)
			//
			// (2) The destination has a unique index.  (The xfer optimization is unable to test uniqueness.)
			//
			// (3) onError is something other than OE_Abort and OE_Rollback.
			addr1 = v->AddOp2(OP_Rewind, destId, 0);
			emptyDestTest = v->AddOp2(OP_Goto, 0, 0);
			v->JumpHere(addr1);
		}
		else
			emptyDestTest = 0;
		Insert::OpenTable(parse, srcId, dbSrcId, src, OP_OpenRead);
		int emptySrcTest = v->AddOp2(OP_Rewind, srcId, 0); // Address of test for empty pSrc
		int regData = Expr::GetTempReg(parse); // Registers holding data and rowid
		int regRowid = Expr::GetTempReg(parse); // Registers holding data and rowid
		if (dest->PKey >= 0)
		{
			addr1 = v->AddOp2(OP_Rowid, srcId, regRowid);
			int addr2 = v->AddOp3(OP_NotExists, destId, 0, regRowid); // Loop addresses
			parse->HaltConstraint(RC_CONSTRAINT_PRIMARYKEY, onError, "PRIMARY KEY must be unique", Vdbe::P4T_STATIC);
			v->JumpHere(addr2);
			AutoIncStep(parse, regAutoinc, regRowid);
		}
		else if (!dest->Index)
			addr1 = v->AddOp2(OP_NewRowid, destId, regRowid);
		else
		{
			addr1 = v->AddOp2(OP_Rowid, srcId, regRowid);
			_assert((dest->TabFlags & TF_Autoincrement) == 0);
		}
		v->AddOp2(OP_RowData, srcId, regData);
		v->AddOp3(OP_Insert, destId, regData, regRowid);
		v->ChangeP5(Vdbe::OPFLAG_NCHANGE|Vdbe::OPFLAG_LASTROWID|Vdbe::OPFLAG_APPEND);
		v->ChangeP4(-1, dest->Name, 0);
		v->AddOp2(OP_Next, srcId, addr1);
		for (destIdx = dest->Index; destIdx; destIdx = destIdx->Next)
		{
			for (srcIdx = src->Index; _ALWAYS(srcIdx); srcIdx = srcIdx->Next)
				if (XferCompatibleIndex(destIdx, srcIdx)) break;
			_assert(srcIdx);
			v->AddOp2(OP_Close, srcId, 0);
			v->AddOp2(OP_Close, destId, 0);
			KeyInfo *key = parse->IndexKeyinfo(srcIdx); // Key information for an index
			v->AddOp4(OP_OpenRead, srcId, srcIdx->Id, dbSrcId, (char *)key, Vdbe::P4T_KEYINFO_HANDOFF);
			v->Comment("%s", srcIdx->Name);
			key = parse->IndexKeyinfo(destIdx);
			v->AddOp4(OP_OpenWrite, destId, destIdx->Id, dbDestId, (char *)key, Vdbe::P4T_KEYINFO_HANDOFF);
			v->Comment("%s", destIdx->Name);
			addr1 = v->AddOp2(OP_Rewind, srcId, 0);
			v->AddOp2(OP_RowKey, srcId, regData);
			v->AddOp3(OP_IdxInsert, destId, regData, 1);
			v->AddOp2(OP_Next, srcId, addr1+1);
			v->JumpHere(addr1);
		}
		v->JumpHere(emptySrcTest);
		Expr::ReleaseTempReg(parse, regRowid);
		Expr::ReleaseTempReg(parse, regData);
		v->AddOp2(OP_Close, srcId, 0);
		v->AddOp2(OP_Close, destId, 0);
		if (emptyDestTest)
		{
			v->AddOp2(OP_Halt, RC_OK, 0);
			v->JumpHere(emptyDestTest);
			v->AddOp2(OP_Close, destId, 0);
			return false;
		}
		return true;
	}
#endif

}}