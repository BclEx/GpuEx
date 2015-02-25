// delete.c
#include "..\Core+Vdbe.cu.h"

namespace Core { namespace Command
{
	__device__ Table *Delete::SrcListLookup(Parse *parse, SrcList *src)
	{
		SrcList::SrcListItem *item = src->Ids;
		_assert(item && src->Srcs == 1);
		Table *table = parse->LocateTableItem(false, item);
		Parse::DeleteTable(parse->Ctx, item->Table);
		item->Table = table;
		if (table)
			table->Refs++;
		if (Select::IndexedByLookup(parse, item))
			table = nullptr;
		return table;
	}

	__device__ bool Delete::IsReadOnly(Parse *parse, Table *table, bool viewOk)
	{
		// A table is not writable under the following circumstances:
		//   1) It is a virtual table and no implementation of the xUpdate method has been provided, or
		//   2) It is a system table (i.e. sqlite_master), this call is not part of a nested parse and writable_schema pragma has not 
		//      been specified.
		// In either case leave an error message in pParse and return non-zero.
		if ((IsVirtual(table) && VTable::GetVTable(parse->Ctx, table)->Module->IModule->Update == nullptr) ||
			((table->TabFlags & TF_Readonly) != 0 && (parse->Ctx->Flags & Context::FLAG_WriteSchema) == 0 && parse->Nested == 0))
		{
			parse->ErrorMsg("table %s may not be modified", table->Name);
			return true;
		}

#ifndef OMIT_VIEW
		if (!viewOk && table->Select)
		{
			parse->ErrorMsg("cannot modify %s because it is a view", table->Name);
			return true;
		}
#endif
		return false;
	}

#if !defined(OMIT_VIEW) && !defined(OMIT_TRIGGER)
	__device__ void Delete::MaterializeView(Parse *parse, Table *view, Expr *where_, int curId)
	{
		Context *ctx = parse->Ctx;
		int db = Prepare::SchemaToIndex(ctx, view->Schema);

		where_ = Expr::Dup(ctx, where_, 0);
		SrcList *from = Parse::SrcListAppend(ctx, nullptr, nullptr, nullptr);

		if (from)
		{
			_assert(from->Srcs == 1);
			from->Ids[0].Name = _tagstrdup(ctx, view->Name);
			from->Ids[0].Database = _tagstrdup(ctx, ctx->DBs[db].Name);
			_assert(!from->Ids[0].On);
			_assert(!from->Ids[0].Using);
		}

		Select *select = Select::New(parse, nullptr, from, where_, nullptr, nullptr, nullptr, (SF)0, nullptr, nullptr);
		if (select) select->SelFlags |= SF_Materialize;

		SelectDest dest;
		Select::DestInit(&dest, SRT_EphemTab, curId);
		Select::Select_(parse, select, &dest);
		Select::Delete(ctx, select);
	}
#endif

#if 1 || defined(ENABLE_UPDATE_DELETE_LIMIT) && !defined(OMIT_SUBQUERY)
	__device__ Expr *Delete::LimitWhere(Parse *parse, SrcList *src, Expr *where_, ExprList *orderBy, Expr *limit, Expr *offset, char *stmtType)
	{
		// Check that there isn't an ORDER BY without a LIMIT clause.
		if (orderBy && (limit == 0))
		{
			parse->ErrorMsg("ORDER BY without LIMIT on %s", stmtType);
			goto limit_where_cleanup_2;
		}

		// We only need to generate a select expression if there is a limit/offset term to enforce.
		if (!limit)
		{
			_assert(!offset); // if pLimit is null, pOffset will always be null as well. 
			return where_;
		}

		// Generate a select expression tree to enforce the limit/offset term for the DELETE or UPDATE statement.  For example:
		//   DELETE FROM table_a WHERE col1=1 ORDER BY col2 LIMIT 1 OFFSET 1
		// becomes:
		//   DELETE FROM table_a WHERE rowid IN ( 
		//     SELECT rowid FROM table_a WHERE col1=1 ORDER BY col2 LIMIT 1 OFFSET 1
		//   );
		Expr *selectRowid = Expr::PExpr_(parse, TK_ROW, 0, 0, 0); // SELECT rowid ...
		if (!selectRowid) goto limit_where_cleanup_2;
		ExprList *list = Expr::ListAppend(parse, 0, selectRowid);
		if (!list) goto limit_where_cleanup_2; // Expression list contaning only pSelectRowid

		// duplicate the FROM clause as it is needed by both the DELETE/UPDATE tree and the SELECT subtree.
		SrcList *selectSrc = Expr::SrcListDup(parse->Ctx, src, 0); // SELECT rowid FROM x ... (dup of pSrc)
		if (!selectSrc)
		{
			Expr::ListDelete(parse->Ctx, list);
			goto limit_where_cleanup_2;
		}

		// generate the SELECT expression tree.
		Select *select = Select::New(parse, list, selectSrc, where_, 0, 0, orderBy, (SF)0, limit, offset); // Complete SELECT tree
		if (!select) return nullptr;

		// now generate the new WHERE rowid IN clause for the DELETE/UDPATE
		Expr *whereRowid = Expr::PExpr_(parse, TK_ROW, 0, 0, 0); // WHERE rowid ..
		if (!whereRowid) goto limit_where_cleanup_1;
		Expr *inClause = Expr::PExpr_(parse, TK_IN, whereRowid, 0, 0); // WHERE rowid IN ( select )
		if (!inClause) goto limit_where_cleanup_1;

		inClause->x.Select = select;
		inClause->Flags |= EP_xIsSelect;
		inClause->SetHeight(parse);
		return inClause;

		// something went wrong. clean up anything allocated.
limit_where_cleanup_1:
		Select::Delete(parse->Ctx, select);
		return nullptr;

limit_where_cleanup_2:
		Expr::Delete(parse->Ctx, where_);
		Expr::ListDelete(parse->Ctx, orderBy);
		Expr::Delete(parse->Ctx, limit);
		Expr::Delete(parse->Ctx, offset);
		return nullptr;
	}
#endif

	__device__ void Delete::DeleteFrom(Parse *parse, SrcList *tabList, Expr *where_)
	{
		AuthContext sContext; // Authorization context
		_memset(&sContext, 0, sizeof(sContext));
		Context *ctx = parse->Ctx; // Main database structure
		if (parse->Errs || ctx->MallocFailed)
			goto delete_from_cleanup;
		_assert(tabList->Srcs == 1);

		// Locate the table which we want to delete.  This table has to be put in an SrcList structure because some of the subroutines we
		// will be calling are designed to work with multiple tables and expect an SrcList* parameter instead of just a Table* parameter.
		Table *table = SrcListLookup(parse, tabList); // The table from which records will be deleted
		if (!table) goto delete_from_cleanup;

		// Figure out if we have any triggers and if the table being deleted from is a view
#ifndef OMIT_TRIGGER
		Trigger *trigger = Trigger::TriggersExist(parse, table, TK_DELETE, 0, 0); // List of table triggers, if required
#ifdef OMIT_VIEW
#define isView false
#else
		bool isView = (table->Select != nullptr); // True if attempting to delete from a view
#endif
#else
#define trigger nullptr
#define isView false
#endif

		// If table is really a view, make sure it has been initialized.
		if (parse->ViewGetColumnNames(table) || IsReadOnly(parse, table, (trigger != nullptr)))
			goto delete_from_cleanup;
		int db = Prepare::SchemaToIndex(ctx, table->Schema); // Database number
		_assert(db < ctx->DBs.length);
		const char *dbName = ctx->DBs[db].Name; // Name of database holding pTab
		ARC rcauth = Auth::Check(parse, AUTH_DELETE, table->Name, 0, dbName); // Value returned by authorization callback
		assert(rcauth == ARC_OK || rcauth == ARC_DENY || rcauth == ARC_IGNORE );
		if (rcauth == ARC_DENY)
			goto delete_from_cleanup;
		_assert(!isView || trigger);

		// Assign cursor number to the table and all its indices.
		_assert(tabList->Srcs == 1);
		int curId = tabList->Ids[0].Cursor = parse->Tabs++; // VDBE Cursor number for pTab
		Index *idx; // For looping over indices of the table
		for (idx = table->Index; idx; idx = idx->Next)
			parse->Tabs++;

		// Start the view context
		if (isView)
			Auth::ContextPush(parse, &sContext, table->Name);

		// Begin generating code.
		Vdbe *v = parse->GetVdbe(); // The virtual database engine
		if (!v)
			goto delete_from_cleanup;
		if (parse->Nested == 0) v->CountChanges();
		parse->BeginWriteOperation(1, db);

		// If we are trying to delete from a view, realize that view into a ephemeral table.
#if !defined(OMIT_VIEW) && !defined(OMIT_TRIGGER)
		if (isView)
			MaterializeView(parse, table, where_, curId);
#endif

		// Resolve the column names in the WHERE clause.
		NameContext sNC; // Name context to resolve expressions in
		_memset(&sNC, 0, sizeof(sNC));
		sNC.Parse = parse;
		sNC.SrcList = tabList;
		if (Walker::ResolveExprNames(&sNC, where_))
			goto delete_from_cleanup;

		// Initialize the counter of the number of rows deleted, if we are counting rows.
		int memCnt = -1; // Memory cell used for change counting
		if (ctx->Flags & Context::FLAG_CountRows)
		{
			memCnt = ++parse->Mems;
			v->AddOp2(OP_Integer, 0, memCnt);
		}

#ifndef OMIT_TRUNCATE_OPTIMIZATION
		// Special case: A DELETE without a WHERE clause deletes everything. It is easier just to erase the whole table. Prior to version 3.6.5,
		// this optimization caused the row change count (the value returned by API function sqlite3_count_changes) to be set incorrectly.
		if (rcauth == ARC_OK && !where_ && !trigger && !IsVirtual(table) && !parse->FKRequired(table, 0, 0))
		{
			_assert(!isView);
			v->AddOp4(OP_Clear, table->Id, db, memCnt, table->Name, Vdbe::P4T_STATIC);
			for (idx = table->Index; idx; idx = idx->Next)
			{
				_assert(idx->Schema == table->Schema);
				v->AddOp2(OP_Clear, idx->Id, db);
			}
		}
		else
#endif
			// The usual case: There is a WHERE clause so we have to scan through the table and pick which records to delete.
		{
			int rowSet = ++parse->Mems; // Register for rowset of rows to delete
			int rowid = ++parse->Mems; // Used for storing rowid values.

			// Collect rowids of every row to be deleted.
			v->AddOp2(OP_Null, 0, rowSet);
			WhereInfo *winfo = WhereInfo::Begin(parse, tabList, where_, 0, nullptr, WHERE_DUPLICATES_OK, 0); // Information about the WHERE clause
			if (!winfo) goto delete_from_cleanup;
			int regRowid = Expr::CodeGetColumn(parse, table, -1, curId, rowid, 0); // Actual register containing rowids
			v->AddOp2(OP_RowSetAdd, rowSet, regRowid);
			if (ctx->Flags & Context::FLAG_CountRows)
				v->AddOp2(OP_AddImm, memCnt, 1);
			WhereInfo::End(winfo);

			// Delete every item whose key was written to the list during the database scan.  We have to delete items after the scan is complete
			// because deleting an item can change the scan order.
			int end = v->MakeLabel();

			// Unless this is a view, open cursors for the table we are deleting from and all its indices. If this is a view, then the
			// only effect this statement has is to fire the INSTEAD OF triggers.
			if (!isView)
				Insert::OpenTableAndIndices(parse, table, curId, OP_OpenWrite);
			int addr = v->AddOp3(OP_RowSetRead, rowSet, end, rowid);

			// Delete the row
#ifndef OMIT_VIRTUALTABLE
			if (IsVirtual(table))
			{
				const char *vtable = (const char *)VTable::GetVTable(ctx, table);
				VTable::MakeWritable(parse, table);
				v->AddOp4(OP_VUpdate, 0, 1, rowid, vtable, Vdbe::P4T_VTAB);
				v->ChangeP5(OE_Abort);
				parse->MayAbort();
			}
			else
#endif
			{
				bool count = (!parse->Nested); // True to count changes
				GenerateRowDelete(parse, table, curId, rowid, count, trigger, OE_Default);
			}

			// End of the delete loop
			v->AddOp2(OP_Goto, 0, addr);
			v->ResolveLabel(end);

			// Close the cursors open on the table and its indexes.
			if (!isView && !IsVirtual(table))
			{
				int i;
				for (i = 1, idx = table->Index; idx; i++, idx = idx->Next)
					v->AddOp2(OP_Close, curId + i, idx->Id);
				v->AddOp1(OP_Close, curId);
			}
		}

		// Update the sqlite_sequence table by storing the content of the maximum rowid counter values recorded while inserting into
		// autoincrement tables.
		if (!parse->Nested && !parse->TriggerTab)
			Insert::AutoincrementEnd(parse);

		// Return the number of rows that were deleted. If this routine is generating code because of a call to sqlite3NestedParse(), do not
		// invoke the callback function.
		if ((ctx->Flags & Context::FLAG_CountRows) && !parse->Nested && !parse->TriggerTab)
		{
			v->AddOp2(OP_ResultRow, memCnt, 1);
			v->SetNumCols(1);
			v->SetColName(0, COLNAME_NAME, "rows deleted", DESTRUCTOR_STATIC);
		}

delete_from_cleanup:
		Auth::ContextPop(&sContext);
		Parse::SrcListDelete(ctx, tabList);
		Expr::Delete(ctx, where_);
		return;
	}
#ifdef isView
#undef isView
#endif
#ifdef trigger
#undef trigger
#endif

	__device__ void Delete::GenerateRowDelete(Parse *parse, Table *table, int curId, int rowid, int count, Trigger *trigger, OE onconf)
	{
		// Vdbe is guaranteed to have been allocated by this stage.
		Vdbe *v = parse->V;
		_assert(v);

		// Seek cursor iCur to the row to delete. If this row no longer exists (this can happen if a trigger program has already deleted it), do
		// not attempt to delete it or fire any DELETE triggers.
		int label = v->MakeLabel(); // Label resolved to end of generated code
		v->AddOp3(OP_NotExists, curId, label, rowid);

		// If there are any triggers to fire, allocate a range of registers to use for the old.* references in the triggers.
		int oldId = 0; // First register in OLD.* array
		if (parse->FKRequired(table, 0, 0) || trigger)
		{
			// TODO: Could use temporary registers here. Also could attempt to avoid copying the contents of the rowid register.
			uint32 mask = trigger->Colmask(parse, nullptr, false, TRIGGER_BEFORE|TRIGGER_AFTER, table, onconf); // Mask of OLD.* columns in use
			mask |= parse->FKOldmask(table);
			oldId = parse->Mems+1;
			parse->Mems += (1 + table->Cols.length);

			// Populate the OLD.* pseudo-table register array. These values will be used by any BEFORE and AFTER triggers that exist.
			v->AddOp2(OP_Copy, rowid, oldId);
			for (int col = 0; col < table->Cols.length; col++) // Iterator used while populating OLD.*
				if (mask == 0xffffffff || mask & (1<<col))
					Expr::CodeGetColumnOfTable(v, table, curId, col, oldId + col+1);

			// Invoke BEFORE DELETE trigger programs.
			trigger->CodeRowTrigger(parse, TK_DELETE, nullptr, TRIGGER_BEFORE, table, oldId, onconf, label);

			// Seek the cursor to the row to be deleted again. It may be that the BEFORE triggers coded above have already removed the row
			// being deleted. Do not attempt to delete the row a second time, and do not fire AFTER triggers.
			v->AddOp3(OP_NotExists, curId, label, rowid);

			// Do FK processing. This call checks that any FK constraints that refer to this table (i.e. constraints attached to other tables) are not violated by deleting this row.
			parse->FKCheck(table, oldId, 0);
		}

		// Delete the index and table entries. Skip this step if table is really a view (in which case the only effect of the DELETE statement is to fire the INSTEAD OF triggers).
		if (!table->Select)
		{
			GenerateRowIndexDelete(parse, table, curId, nullptr);
			v->AddOp2(OP_Delete, curId, (count ? Vdbe::OPFLAG_NCHANGE : 0));
			if (count)
				v->ChangeP4(-1, table->Name, Vdbe::P4T_TRANSIENT);
		}

		// Do any ON CASCADE, SET NULL or SET DEFAULT operations required to handle rows (possibly in other tables) that refer via a foreign key to the row just deleted.
		parse->FKActions(table, 0, oldId);

		// Invoke AFTER DELETE trigger programs.
		trigger->CodeRowTrigger(parse, TK_DELETE, 0, TRIGGER_AFTER, table, oldId, onconf, label);

		// Jump here if the row had already been deleted before any BEFORE trigger programs were invoked. Or if a trigger program throws a RAISE(IGNORE) exception.
		v->ResolveLabel(label);
	}

	__device__ void Delete::GenerateRowIndexDelete(Parse *parse, Table *table, int curId, int *regIdxs)
	{
		int i;
		Index *idx;
		for (i = 1, idx = table->Index; idx; i++, idx = idx->Next)
		{
			if (regIdxs && regIdxs[i-1] == 0) continue;
			int r1 = GenerateIndexKey(parse, idx, curId, 0, false);
			parse->V->AddOp3(OP_IdxDelete, curId+i, r1, idx->Columns.length+1);
		}
	}

	__device__ int Delete::GenerateIndexKey(Parse *parse, Index *index, int curId, int regOut, bool doMakeRec)
	{
		Vdbe *v = parse->V;
		Table *table = index->Table;

		int cols = index->Columns.length;
		int regBase = Expr::GetTempRange(parse, cols+1);
		v->AddOp2(OP_Rowid, curId, regBase+cols);
		for (int j = 0; j < cols; j++)
		{
			int idx = index->Columns[j];
			if (idx == table->PKey)
				v->AddOp2(OP_SCopy, regBase+cols, regBase+j);
			else
			{
				v->AddOp3(OP_Column, curId, idx, regBase+j);
				Update::ColumnDefault(v, table, idx, -1);
			}
		}
		if (doMakeRec)
		{
			const char *affName = (table->Select || CtxOptimizationDisabled(parse->Ctx, OPTFLAG_IdxRealAsInt) ? nullptr : Insert::IndexAffinityStr(v, index));
			v->AddOp3(OP_MakeRecord, regBase, cols+1, regOut);
			v->ChangeP4(-1, affName, Vdbe::P4T_TRANSIENT);
		}
		Expr::ReleaseTempRange(parse, regBase, cols+1);
		return regBase;
	}

} }