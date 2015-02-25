#include "..\Core+Vdbe.cu.h"

namespace Core { namespace Command
{

#ifndef OMIT_VIRTUALTABLE
	__device__ static void UpdateVirtualTable(Parse *parse, SrcList *src, Table *table, ExprList *changes, Expr *rowid, int *xrefs, Expr *where_, OE onError);
#endif

	__device__ void Update::ColumnDefault(Vdbe *v, Table *table, int i, int regId)
	{
		_assert(table);
		if (!table->Select)
		{
			TEXTENCODE encode = CTXENCODE(v->Ctx);
			Column *col = &table->Cols[i];
			v->Comment("%s.%s", table->Name, col->Name);
			_assert(i < table->Cols.length);
			Mem *value;
			Vdbe::ValueFromExpr(v->Ctx, col->Dflt, encode, col->Affinity, &value);
			if (value)
				v->ChangeP4(-1, (const char *)value, Vdbe::P4T_MEM);
#ifndef OMIT_FLOATING_POINT
			if (regId >= 0 && table->Cols[i].Affinity == AFF_REAL)
				v->AddOp1(OP_RealAffinity, regId);
#endif
		}
	}

	__device__ void Update::Update_(Parse *parse, SrcList *tabList, ExprList *changes, Expr *where_, OE onError)
	{
		int i, j;              // Loop counters

		AuthContext sContext;  // The authorization context
		_memset(&sContext, 0, sizeof(sContext));
		Context *ctx = parse->Ctx; // The database structure
		if (parse->Errs || ctx->MallocFailed)
			goto update_cleanup;
		_assert(tabList->Srcs == 1);

		// Locate the table which we want to update. 
		Table *table = Delete::SrcListLookup(parse, tabList); // The table to be updated
		if (!table) goto update_cleanup;
		int db = Prepare::SchemaToIndex(ctx, table->Schema); // Database containing the table being updated

		// Figure out if we have any triggers and if the table being updated is a view.
#ifndef OMIT_TRIGGER
		TRIGGER tmask; // Mask of TRIGGER_BEFORE|TRIGGER_AFTER
		Trigger *trigger = Trigger::TriggersExist(parse, table, TK_UPDATE, changes, &tmask); // List of triggers on table, if required
#ifdef OMIT_VIEW
#define isView false
#else
		bool isView = (table->Select != nullptr); // True when updating a view (INSTEAD OF trigger)
#endif		
		_assert(trigger || tmask == 0);
#else
#define tmask 0
#define trigger nullptr
#define isView false
#endif
		if (parse->ViewGetColumnNames(table) || Delete::IsReadOnly(parse, table, tmask != 0))
			goto update_cleanup;
		int *xrefs = (int *)_tagalloc(ctx, sizeof(int) * table->Cols.length); // xrefs[i] is the index in pChanges->a[] of the an expression for the i-th column of the table. xrefs[i]==-1 if the i-th column is not changed.
		if (!xrefs) goto update_cleanup;
		for (i = 0; i < table->Cols.length; i++) xrefs[i] = -1;

		// Allocate a cursors for the main database table and for all indices. The index cursors might not be used, but if they are used they
		// need to occur right after the database cursor.  So go ahead and allocate enough space, just in case.
		int curId; // VDBE Cursor number of table
		tabList->Ids[0].Cursor = curId = parse->Tabs++;
		Index *idx; // For looping over indices
		for (idx = table->Index; idx; idx = idx->Next)
			parse->Tabs++;

		// Initialize the name-context
		NameContext sNC; // The name-context to resolve expressions in
		_memset(&sNC, 0, sizeof(sNC));
		sNC.Parse = parse;
		sNC.SrcList = tabList;

		// Resolve the column names in all the expressions of the of the UPDATE statement.  Also find the column index
		// for each column to be updated in the pChanges array.  For each column to be updated, make sure we have authorization to change that column.
		bool chngRowid = false; // True if the record number is being changed
		Expr *rowidExpr = nullptr; // Expression defining the new record number
		for (i = 0; i < changes->Exprs; i++)
		{
			if (Walker::ResolveExprNames(&sNC, changes->Ids[i].Expr))
				goto update_cleanup;
			for (j = 0; j < table->Cols.length; j++)
			{
				if (!_strcmp(table->Cols[j].Name, changes->Ids[i].Name))
				{
					if (j == table->PKey)
					{
						chngRowid = true;
						rowidExpr = changes->Ids[i].Expr;
					}
					xrefs[j] = i;
					break;
				}
			}
			if (j >= table->Cols.length)
			{
				if (Expr::IsRowid(changes->Ids[i].Name))
				{
					chngRowid = true;
					rowidExpr = changes->Ids[i].Expr;
				}
				else
				{
					parse->ErrorMsg("no such column: %s", changes->Ids[i].Name);
					parse->CheckSchema = 1;
					goto update_cleanup;
				}
			}
#ifndef OMIT_AUTHORIZATION
			{
				ARC rc = Auth::Check(parse, AUTH_UPDATE, table->Name, table->Cols[j].Name, ctx->DBs[db].Name);
				if (rc == ARC_DENY) goto update_cleanup;
				else if (rc == ARC_IGNORE) xrefs[j] = -1;
			}
#endif
		}

		bool hasFK = parse->FKRequired(table, xrefs, chngRowid); // True if foreign key processing is required

		// Allocate memory for the array aRegIdx[].  There is one entry in the array for each index associated with table being updated.  Fill in
		// the value with a register number for indices that are to be used and with zero for unused indices.
		int idxLength; // Number of indices that need updating
		for (idxLength = 0, idx = table->Index; idx; idx = idx->Next, idxLength++) ;
		int *regIdxs = nullptr; // One register assigned to each index to be updated
		if (idxLength > 0)
		{
			regIdxs = (int *)_tagalloc(ctx, sizeof(Index *) * idxLength);
			if (!regIdxs) goto update_cleanup;
		}
		for (j = 0, idx = table->Index; idx; idx = idx->Next, j++)
		{
			int regId;
			if (hasFK || chngRowid)
				regId = ++parse->Mems;
			else
			{
				regId = 0;
				for (i = 0; i < idx->Columns.length; i++)
					if (xrefs[idx->Columns[i]] >= 0)
					{
						regId = ++parse->Mems;
						break;
					}
			}
			regIdxs[j] = regId;
		}

		// Begin generating code.
		Vdbe *v = parse->GetVdbe(); // The virtual database engine
		if (!v) goto update_cleanup;
		if (parse->Nested == 0) v->CountChanges();
		parse->BeginWriteOperation(1, db);

#ifndef OMIT_VIRTUALTABLE
		// Virtual tables must be handled separately
		if (IsVirtual(table))
		{
			UpdateVirtualTable(parse, tabList, table, changes, rowidExpr, xrefs, where_, onError);
			where_ = nullptr;
			tabList = nullptr;
			goto update_cleanup;
		}
#endif

		// Register Allocations
		int regRowCount = 0;   // A count of rows changed
		int regOldRowid;       // The old rowid
		int regNewRowid;       // The new rowid
		int regNew;            // Content of the NEW.* table in triggers
		int regOld = 0;        // Content of OLD.* table in triggers
		int regRowSet = 0;     // Rowset of rows to be updated

		// Allocate required registers.
		regRowSet = ++parse->Mems;
		regOldRowid = regNewRowid = ++parse->Mems;
		if (trigger || hasFK)
		{
			regOld = parse->Mems + 1;
			parse->Mems += table->Cols.length;
		}
		if (chngRowid || trigger || hasFK)
			regNewRowid = ++parse->Mems;
		regNew = parse->Mems + 1;
		parse->Mems += table->Cols.length;

		// Start the view context.
		if (isView)
			Auth::ContextPush(parse, &sContext, table->Name);

		// If we are trying to update a view, realize that view into a ephemeral table.
#if !defined(OMIT_VIEW) && !defined(OMIT_TRIGGER)
		if (isView)
			Delete::MaterializeView(parse, table, where_, curId);
#endif

		// Resolve the column names in all the expressions in the WHERE clause.
		if (Walker::ResolveExprNames(&sNC, where_))
			goto update_cleanup;

		// Begin the database scan
		v->AddOp3(OP_Null, 0, regRowSet, regOldRowid);
		WhereInfo *winfo = WhereInfo::Begin(parse, tabList, where_, 0, nullptr, WHERE_ONEPASS_DESIRED, 0); // Information about the WHERE clause
		if (!winfo) goto update_cleanup;
		bool okOnePass = winfo->OkOnePass; // True for one-pass algorithm without the FIFO

		// Remember the rowid of every item to be updated.
		v->AddOp2(OP_Rowid, curId, regOldRowid);
		if (!okOnePass)
			v->AddOp2(OP_RowSetAdd, regRowSet, regOldRowid);

		// End the database scan loop.
		WhereInfo::End(winfo);

		// Initialize the count of updated rows
		if ((ctx->Flags & Context::FLAG_CountRows) && !parse->TriggerTab)
		{
			regRowCount = ++parse->Mems;
			v->AddOp2(OP_Integer, 0, regRowCount);
		}

		bool openAll = false; // True if all indices need to be opened
		if (!isView)
		{
			// Open every index that needs updating.  Note that if any index could potentially invoke a REPLACE conflict resolution 
			// action, then we need to open all indices because we might need to be deleting some records.
			if (!okOnePass) Insert::OpenTable(parse, curId, db, table, OP_OpenWrite); 
			if (onError == OE_Replace)
				openAll = true;
			else
			{
				openAll = false;
				for (idx = table->Index; idx; idx = idx->Next)
				{
					if (idx->OnError == OE_Replace)
					{
						openAll = true;
						break;
					}
				}
			}
			for (i = 0, idx = table->Index; idx; idx = idx->Next, i++)
			{
				_assert(regIdxs);
				if (openAll || regIdxs[i] > 0)
				{
					KeyInfo *key = parse->IndexKeyinfo(idx);
					v->AddOp4(OP_OpenWrite, curId+i+1, idx->Id, db, (char *)key, Vdbe::P4T_KEYINFO_HANDOFF);
					_assert(parse->Tabs > curId+i+1);
				}
			}
		}

		// Top of the update loop
		int addr = 0; // VDBE instruction address of the start of the loop
		if (okOnePass)
		{
			int a1 = v->AddOp1(OP_NotNull, regOldRowid);
			addr = v->AddOp0(OP_Goto);
			v->JumpHere(a1);
		}
		else
			addr = v->AddOp3(OP_RowSetRead, regRowSet, 0, regOldRowid);

		// Make cursor iCur point to the record that is being updated. If this record does not exist for some reason (deleted by a trigger,
		// for example, then jump to the next iteration of the RowSet loop.
		v->AddOp3(OP_NotExists, curId, addr, regOldRowid);

		// If the record number will change, set register regNewRowid to contain the new value. If the record number is not being modified,
		// then regNewRowid is the same register as regOldRowid, which is already populated.
		_assert(chngRowid || trigger || hasFK || regOldRowid == regNewRowid);
		if (chngRowid)
		{
			Expr::Code(parse, rowidExpr, regNewRowid);
			v->AddOp1(OP_MustBeInt, regNewRowid);
		}

		// If there are triggers on this table, populate an array of registers with the required old.* column data.
		if (hasFK || trigger)
		{
			uint32 oldmask = (hasFK ? parse->FKOldmask(table) : 0);
			oldmask |= trigger->Colmask(parse, changes, 0, TRIGGER_BEFORE|TRIGGER_AFTER, table, onError);
			for (i = 0; i < table->Cols.length; i++)
			{
				if (xrefs[i] < 0 || oldmask == 0xffffffff || (i < 32 && (oldmask & (1<<i))))
					Expr::CodeGetColumnOfTable(v, table, curId, i, regOld+i);
				else
					v->AddOp2(OP_Null, 0, regOld+i);
			}
			if (!chngRowid)
				v->AddOp2(OP_Copy, regOldRowid, regNewRowid);
		}

		// Populate the array of registers beginning at regNew with the new row data. This array is used to check constaints, create the new
		// table and index records, and as the values for any new.* references made by triggers.
		//
		// If there are one or more BEFORE triggers, then do not populate the registers associated with columns that are (a) not modified by
		// this UPDATE statement and (b) not accessed by new.* references. The values for registers not modified by the UPDATE must be reloaded from 
		// the database after the BEFORE triggers are fired anyway (as the trigger may have modified them). So not loading those that are not going to
		// be used eliminates some redundant opcodes.
		int newmask = trigger->Colmask(parse, changes, 1, TRIGGER_BEFORE, table, onError); // Mask of NEW.* columns accessed by BEFORE triggers
		v->AddOp3(OP_Null, 0, regNew, regNew+table->Cols.length-1);
		for (i = 0; i < table->Cols.length; i++)
		{
			if (i == table->PKey)
			{
				//!v->AddOp2(OP_Null, 0, regNew + i);
			}
			else
			{
				j = xrefs[i];
				if (j >= 0)
					Expr::Code(parse, changes->Ids[j].Expr, regNew+i);
				else if ((tmask & TRIGGER_BEFORE) == 0 || i > 31 || (newmask & (1<<i)))
				{
					// This branch loads the value of a column that will not be changed into a register. This is done if there are no BEFORE triggers, or
					// if there are one or more BEFORE triggers that use this value via a new.* reference in a trigger program.
					ASSERTCOVERAGE(i == 31);
					ASSERTCOVERAGE(i == 32);
					v->AddOp3(OP_Column, curId, i, regNew+i);
					ColumnDefault(v, table, i, regNew+i);
				}
			}
		}

		// Fire any BEFORE UPDATE triggers. This happens before constraints are verified. One could argue that this is wrong.
		if (tmask & TRIGGER_BEFORE)
		{
			v->AddOp2(OP_Affinity, regNew, table->Cols.length);
			Insert::TableAffinityStr(v, table);
			trigger->CodeRowTrigger(parse, TK_UPDATE, changes, TRIGGER_BEFORE, table, regOldRowid, onError, addr);

			// The row-trigger may have deleted the row being updated. In this case, jump to the next row. No updates or AFTER triggers are 
			// required. This behavior - what happens when the row being updated is deleted or renamed by a BEFORE trigger - is left undefined in the documentation.
			v->AddOp3(OP_NotExists, curId, addr, regOldRowid);

			// If it did not delete it, the row-trigger may still have modified some of the columns of the row being updated. Load the values for 
			// all columns not modified by the update statement into their registers in case this has happened.
			for (i = 0; i < table->Cols.length; i++)
				if (xrefs[i] < 0 && i != table->PKey)
				{
					v->AddOp3(OP_Column, curId, i, regNew+i);
					ColumnDefault(v, table, i, regNew+i);
				}
		}

		if (!isView)
		{
			// Do constraint checks.
			Insert::GenerateConstraintChecks(parse, table, curId, regNewRowid, regIdxs, (chngRowid ? regOldRowid : 0), 1, onError, addr, 0);

			// Do FK constraint checks.
			if (hasFK)
				parse->FKCheck(table, regOldRowid, 0);

			// Delete the index entries associated with the current record.
			int j1 = v->AddOp3(OP_NotExists, curId, 0, regOldRowid);  // Address of jump instruction
			Delete::GenerateRowIndexDelete(parse, table, curId, regIdxs);

			// If changing the record number, delete the old record.
			if (hasFK || chngRowid)
				v->AddOp2(OP_Delete, curId, 0);
			v->JumpHere(j1);

			if (hasFK)
				parse->FKCheck(table, 0, regNewRowid);

			// Insert the new index entries and the new record.
			Insert::CompleteInsertion(parse, table, curId, regNewRowid, regIdxs, 1, 0, 0);

			// Do any ON CASCADE, SET NULL or SET DEFAULT operations required to handle rows (possibly in other tables) that refer via a foreign key
			// to the row just updated.
			if (hasFK)
				parse->FKActions(table, changes, regOldRowid);
		}

		// Increment the row counter 
		if ((ctx->Flags & Context::FLAG_CountRows) && !parse->TriggerTab)
			v->AddOp2(OP_AddImm, regRowCount, 1);

		trigger->CodeRowTrigger(parse, TK_UPDATE, changes, TRIGGER_AFTER, table, regOldRowid, onError, addr);

		// Repeat the above with the next record to be updated, until all record selected by the WHERE clause have been updated.
		v->AddOp2(OP_Goto, 0, addr);
		v->JumpHere(addr);

		// Close all tables
		_assert(regIdxs);
		for (i = 0, idx = table->Index; idx; idx = idx->Next, i++)
			if (openAll || regIdxs[i] > 0)
				v->AddOp2(OP_Close, curId+i+1, 0);
		v->AddOp2(OP_Close, curId, 0);

		// Update the sqlite_sequence table by storing the content of the maximum rowid counter values recorded while inserting into
		// autoincrement tables.
		if (parse->Nested == 0 && parse->TriggerTab == 0)
			Insert::AutoincrementEnd(parse);

		// Return the number of rows that were changed. If this routine is generating code because of a call to sqlite3NestedParse(), do not
		// invoke the callback function.
		if ((ctx->Flags & Context::FLAG_CountRows) && !parse->TriggerTab && !parse->Nested)
		{
			v->AddOp2(OP_ResultRow, regRowCount, 1);
			v->SetNumCols(1);
			v->SetColName(0, COLNAME_NAME, "rows updated", DESTRUCTOR_STATIC);
		}

update_cleanup:
		#if !OMIT_AUTHORIZATION
		Auth::ContextPop(&sContext);
		#endif
		_tagfree(ctx, regIdxs);
		_tagfree(ctx, xrefs);
		Parse::SrcListDelete(ctx, tabList);
		Expr::ListDelete(ctx, changes);
		Expr::Delete(ctx, where_);
		return;
	}
	// Make sure "isView" and other macros defined above are undefined. Otherwise thely may interfere with compilation of other functions in this file
	// (or in another file, if this file becomes part of the amalgamation).
#ifdef isView
#undef isView
#endif
#ifdef trigger
#undef trigger
#endif

#ifndef OMIT_VIRTUALTABLE
	__device__ static void UpdateVirtualTable(Parse *parse, SrcList *src, Table *table, ExprList *changes, Expr *rowid, int *xrefs, Expr *where_, OE onError)
	{
		int i;
		Context *ctx = parse->Ctx; // Database connection
		const char *vtable = (const char *)VTable::GetVTable(ctx, table);
		SelectDest dest;

		// Construct the SELECT statement that will find the new values for all updated rows. 
		ExprList *list = Expr::ListAppend(parse, nullptr, Expr::Expr_(ctx, TK_ID, "_rowid_")); // The result set of the SELECT statement
		if (rowid)
			list = Expr::ListAppend(parse, list, Expr::Dup(ctx, rowid, 0));
		_assert(table->PKey < 0);
		for (i = 0; i < table->Cols.length; i++)
		{
			Expr *expr = (xrefs[i] >= 0 ? Expr::Dup(ctx, changes->Ids[xrefs[i]].Expr, 0) : Expr::Expr_(ctx, TK_ID, table->Cols[i].Name)); // Temporary expression
			list = Expr::ListAppend(parse, list, expr);
		}
		Select *select = Select::New(parse, list, src, where_, nullptr, nullptr, nullptr, (SF)0, nullptr, nullptr); // The SELECT statement

		// Create the ephemeral table into which the update results will be stored.
		Vdbe *v = parse->V;  // Virtual machine under construction
		_assert(v);
		int ephemTab = parse->Tabs++; // Table holding the result of the SELECT
		v->AddOp2(OP_OpenEphemeral, ephemTab, table->Cols.length + 1 + (rowid != 0));
		v->ChangeP5(Btree::OPEN_UNORDERED);

		// fill the ephemeral table 
		Select::DestInit(&dest, SRT_Table, ephemTab);
		Select::Select_(parse, select, &dest);

		// Generate code to scan the ephemeral table and call VUpdate.
		int regId = ++parse->Mems; // First register in set passed to OP_VUpdate
		parse->Mems += table->Cols.length+1;
		int addr = v->AddOp2(OP_Rewind, ephemTab, 0); // Address of top of loop
		v->AddOp3(OP_Column,  ephemTab, 0, regId);
		v->AddOp3(OP_Column, ephemTab, (rowid ? 1 : 0), regId+1);
		for (i = 0; i < table->Cols.length; i++){
			v->AddOp3(OP_Column, ephemTab, i+1+(rowid!=0), regId+2+i);
		}
		VTable::MakeWritable(parse, table);
		v->AddOp4(OP_VUpdate, 0, table->Cols.length+2, regId, vtable, Vdbe::P4T_VTAB);
		v->ChangeP5(onError == OE_Default ? OE_Abort : onError);
		parse->MayAbort();
		v->AddOp2(OP_Next, ephemTab, addr+1);
		v->JumpHere(addr);
		v->AddOp2(OP_Close, ephemTab, 0);

		// Cleanup
		Select::Delete(ctx, select);  
	}
#endif
} }