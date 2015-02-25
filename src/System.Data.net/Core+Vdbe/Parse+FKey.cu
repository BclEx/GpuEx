#pragma region OMIT_FOREIGN_KEY
#ifndef OMIT_FOREIGN_KEY
#include "Core+Vdbe.cu.h"

namespace Core
{
#ifndef OMIT_TRIGGER

	__device__ int Parse::FKLocateIndex(Table *parent, FKey *fkey, Index **indexOut, int **colsOut)
	{
		int colsLength = fkey->Cols.length; // Number of columns in parent key
		char *key = fkey->Cols[0].Col; // Name of left-most parent key column

		// The caller is responsible for zeroing output parameters.
		_assert(indexOut && *indexOut == nullptr);
		_assert(!colsOut || *colsOut == nullptr);

		// If this is a non-composite (single column) foreign key, check if it maps to the INTEGER PRIMARY KEY of table pParent. If so, leave *ppIdx 
		// and *paiCol set to zero and return early. 
		//
		// Otherwise, for a composite foreign key (more than one column), allocate space for the aiCol array (returned via output parameter *paiCol).
		// Non-composite foreign keys do not require the aiCol array.
		int *cols = nullptr; // Value to return via *paiCol
		if (colsLength == 1)
		{
			// The FK maps to the IPK if any of the following are true:
			//   1) There is an INTEGER PRIMARY KEY column and the FK is implicitly mapped to the primary key of table pParent, or
			//   2) The FK is explicitly mapped to a column declared as INTEGER PRIMARY KEY.
			if (parent->PKey >= 0)
				if (!key || !_strcmp(parent->Cols[parent->PKey].Name, key)) return 0;
		}
		else if (colsOut)
		{
			_assert(colsLength > 1);
			cols = (int *)_tagalloc(Ctx, colsLength*sizeof(int));
			if (!cols) return 1;
			*colsOut = cols;
		}

		Index *index = nullptr; // Value to return via *ppIdx
		for (index = parent->Index; index; index = index->Next)
		{
			if (index->Columns.length == colsLength && index->OnError != OE_None)
			{ 
				// pIdx is a UNIQUE index (or a PRIMARY KEY) and has the right number of columns. If each indexed column corresponds to a foreign key
				// column of pFKey, then this index is a winner.  */
				if (!key)
				{
					// If zKey is NULL, then this foreign key is implicitly mapped to the PRIMARY KEY of table pParent. The PRIMARY KEY index may be 
					// identified by the test (Index.autoIndex==2).
					if (index->AutoIndex == 2)
					{
						if (cols)
							for (int i = 0; i < colsLength; i++) cols[i] = fkey->Cols[i].From;
						break;
					}
				}
				else
				{
					// If zKey is non-NULL, then this foreign key was declared to map to an explicit list of columns in table pParent. Check if this
					// index matches those columns. Also, check that the index uses the default collation sequences for each column.
					int i, j;
					for (i = 0; i < colsLength; i++)
					{
						int col = index->Columns[i]; // Index of column in parent tbl
						// If the index uses a collation sequence that is different from the default collation sequence for the column, this index is
						// unusable. Bail out early in this case.
						char *dfltColl = parent->Cols[col].Coll;// Def. collation for column
						if (!dfltColl)
							dfltColl = "BINARY";
						if (_strcmp(index->CollNames[i], dfltColl)) break;

						char *indexCol = parent->Cols[col].Name; // Name of indexed column
						for (j = 0; j < colsLength; j++)
						{
							if (!_strcmp(fkey->Cols[j].Col, indexCol))
							{
								if (cols) cols[i] = fkey->Cols[j].From;
								break;
							}
						}
						if (j == colsLength) break;
					}
					if (i == colsLength) break; // pIdx is usable
				}
			}
		}

		if (!index)
		{
			if (!DisableTriggers)
				ErrorMsg("foreign key mismatch - \"%w\" referencing \"%w\"", fkey->From->Name, fkey->To);
			_tagfree(Ctx, cols);
			return 1;
		}

		*indexOut = index;
		return 0;
	}

	__device__ static void FKLookupParent(Parse *parse, int db, Table *table, Index *index, FKey *fkey, int *cols, int regDataId, int incr, bool isIgnore)
	{
		Vdbe *v = parse->GetVdbe(); // Vdbe to add code to
		int curId = parse->Tabs - 1; // Cursor number to use
		int okId = v->MakeLabel(); // jump here if parent key found

		// If nIncr is less than zero, then check at runtime if there are any outstanding constraints to resolve. If there are not, there is no need
		// to check if deleting this row resolves any outstanding violations.
		//
		// Check if any of the key columns in the child table row are NULL. If any are, then the constraint is considered satisfied. No need to 
		// search for a matching row in the parent table.
		int i;
		if (incr < 0)
			v->AddOp2(OP_FkIfZero, fkey->IsDeferred, okId);
		for (i = 0; i < fkey->Cols.length; i++)
		{
			int regId = cols[i] + regDataId + 1;
			v->AddOp2(OP_IsNull, regId, okId);
		}

		if (!isIgnore)
		{
			if (!index)
			{
				// If pIdx is NULL, then the parent key is the INTEGER PRIMARY KEY column of the parent table (table table).
				int regTempId = Expr::GetTempReg(parse);

				// Invoke MustBeInt to coerce the child key value to an integer (i.e. apply the affinity of the parent key). If this fails, then there
				// is no matching parent key. Before using MustBeInt, make a copy of the value. Otherwise, the value inserted into the child key column
				// will have INTEGER affinity applied to it, which may not be correct.
				v->AddOp2(OP_SCopy, cols[0]+1+regDataId, regTempId);
				int mustBeIntId = v->AddOp2(OP_MustBeInt, regTempId, 0);// Address of MustBeInt instruction

				// If the parent table is the same as the child table, and we are about to increment the constraint-counter (i.e. this is an INSERT operation),
				// then check if the row being inserted matches itself. If so, do not increment the constraint-counter.
				if (table == fkey->From && incr == 1)
					v->AddOp3(OP_Eq, regDataId, okId, regTempId);

				Insert::OpenTable(parse, curId, db, table, OP_OpenRead);
				v->AddOp3(OP_NotExists, curId, 0, regTempId);
				v->AddOp2(OP_Goto, 0, okId);
				v->JumpHere(v->CurrentAddr()-2);
				v->JumpHere(mustBeIntId);
				Expr::ReleaseTempReg(parse, regTempId);
			}
			else
			{
				int colsLength = fkey->Cols.length;
				int regTempId = Expr::GetTempRange(parse, colsLength);
				int regRecId = Expr::GetTempReg(parse);
				KeyInfo *key = parse->IndexKeyinfo(index);

				v->AddOp3(OP_OpenRead, curId, index->Id, db);
				v->ChangeP4(-1, (char *)key, Vdbe::P4T_KEYINFO_HANDOFF);
				for (i = 0; i < colsLength; i++)
					v->AddOp2(OP_Copy, cols[i]+1+regDataId, regTempId+i);

				// If the parent table is the same as the child table, and we are about to increment the constraint-counter (i.e. this is an INSERT operation),
				// then check if the row being inserted matches itself. If so, do not increment the constraint-counter. 
				//
				// If any of the parent-key values are NULL, then the row cannot match itself. So set JUMPIFNULL to make sure we do the OP_Found if any
				// of the parent-key values are NULL (at this point it is known that none of the child key values are).
				if (table == fkey->From && incr == 1)
				{
					int jumpId = v->CurrentAddr() + colsLength + 1;
					for (i = 0; i < colsLength; i++)
					{
						int childId = cols[i]+1+regDataId;
						int parentId = index->Columns[i]+1+regDataId;
						_assert(cols[i] != table->PKey);
						if (index->Columns[i] == table->PKey)
							parentId = regDataId; // The parent key is a composite key that includes the IPK column
						v->AddOp3(OP_Ne, childId, jumpId, parentId);
						v->ChangeP5(AFF_BIT_JUMPIFNULL);
					}
					v->AddOp2(OP_Goto, 0, okId);
				}

				v->AddOp3(OP_MakeRecord, regTempId, colsLength, regRecId);
				v->ChangeP4(-1, Insert::IndexAffinityStr(v, index), Vdbe::P4T_TRANSIENT);
				v->AddOp4Int(OP_Found, curId, okId, regRecId, 0);

				Expr::ReleaseTempReg(parse, regRecId);
				Expr::ReleaseTempRange(parse, regTempId, colsLength);
			}
		}

		if (!fkey->IsDeferred && !parse->Toplevel && !parse->IsMultiWrite)
		{
			// Special case: If this is an INSERT statement that will insert exactly one row into the table, raise a constraint immediately instead of
			// incrementing a counter. This is necessary as the VM code is being generated for will not open a statement transaction.
			_assert(incr == 1);
			parse->HaltConstraint(RC_CONSTRAINT_FOREIGNKEY, OE_Abort, "foreign key constraint failed", Vdbe::P4T_STATIC);
		}
		else
		{
			if (incr > 0 && !fkey->IsDeferred)
				Parse_Toplevel(parse)->_MayAbort = true;
			v->AddOp2(OP_FkCounter, fkey->IsDeferred, incr);
		}

		v->ResolveLabel(okId);
		v->AddOp1(OP_Close, curId);
	}

	__device__ static void FKScanChildren(Parse *parse, SrcList *src, Table *table, Index *index, FKey *fkey, int *cols, int regDataId, int incr)
	{
		Context *ctx = parse->Ctx; // Database handle
		Vdbe *v = parse->GetVdbe();
		_assert(!index || index->Table == table);
		int fkIfZero = 0; // Address of OP_FkIfZero
		if (incr < 0)
			fkIfZero = v->AddOp2(OP_FkIfZero, fkey->IsDeferred, 0);

		// Create an Expr object representing an SQL expression like:
		//
		//   <parent-key1> = <child-key1> AND <parent-key2> = <child-key2> ...
		//
		// The collation sequence used for the comparison should be that of the parent key columns. The affinity of the parent key column should
		// be applied to each child key value before the comparison takes place.
		Expr *where_ = nullptr; // WHERE clause to scan with
		for (int i = 0; i < fkey->Cols.length; i++)
		{
			int col; // Index of column in child table
			Expr *left = Expr::Expr_(ctx, TK_REGISTER, 0); // Value from parent table row
			if (left)
			{
				// Set the collation sequence and affinity of the LHS of each TK_EQ expression to the parent key column defaults.
				if (index)
				{
					col = index->Columns[i];
					Column *colObj = &table->Cols[col];
					if (table->PKey == col) col = -1;
					left->TableId = regDataId + col + 1;
					left->Aff = colObj->Affinity;
					const char *collName = colObj->Coll;
					if (collName == 0) collName = ctx->DefaultColl->Name;
					left = Expr::AddCollateString(parse, left, collName);
				}
				else
				{
					left->TableId = regDataId;
					left->Aff = AFF_INTEGER;
				}
			}
			col = (cols ? cols[i] : fkey->Cols[0].From);
			_assert(col >= 0);
			const char *colName = fkey->From->Cols[col].Name; // Name of column in child table
			Expr *right = Expr::Expr_(ctx, TK_ID, colName); // Column ref to child table
			Expr *eq = Expr::PExpr_(parse, TK_EQ, left, right, 0); // Expression (left = right)
			where_ = Expr::And(ctx, where_, eq);
		}

		// If the child table is the same as the parent table, and this scan is taking place as part of a DELETE operation (operation D.2), omit the
		// row being deleted from the scan by adding ($rowid != rowid) to the WHERE clause, where $rowid is the rowid of the row being deleted.
		if (table == fkey->From && incr > 0)
		{
			Expr *left = Expr::Expr_(ctx, TK_REGISTER, 0); // Value from parent table row
			Expr *right = Expr::Expr_(ctx, TK_COLUMN, 0); // Column ref to child table
			if (left && right)
			{
				left->TableId = regDataId;
				left->Aff = AFF_INTEGER;
				right->TableId = src->Ids[0].Cursor;
				right->ColumnIdx= -1;
			}
			Expr *eq = Expr::PExpr_(parse, TK_NE, left, right, 0); // Expression (left = right)
			where_ = Expr::And(ctx, where_, eq);
		}

		// Resolve the references in the WHERE clause.
		NameContext nameContext; // Context used to resolve WHERE clause
		_memset(&nameContext, 0, sizeof(NameContext));
		nameContext.SrcList = src;
		nameContext.Parse = parse;
		Walker::ResolveExprNames(&nameContext, where_);

		// Create VDBE to loop through the entries in src that match the WHERE clause. If the constraint is not deferred, throw an exception for
		// each row found. Otherwise, for deferred constraints, increment the deferred constraint counter by incr for each row selected.
		WhereInfo *whereInfo = WhereInfo::Begin(parse, src, where_, nullptr, nullptr, (WHERE)0, 0);  // Context used by sqlite3WhereXXX()
		if (incr > 0 && !fkey->IsDeferred)
			Parse_Toplevel(parse)->_MayAbort = true;
		v->AddOp2(OP_FkCounter, fkey->IsDeferred, incr);
		if (whereInfo)
			WhereInfo::End(whereInfo);

		// Clean up the WHERE clause constructed above.
		Expr::Delete(ctx, where_);
		if (fkIfZero)
			v->JumpHere(fkIfZero);
	}

	__device__ FKey *Parse::FKReferences(Table *table)
	{
		int nameLength = _strlen30(table->Name);
		return (FKey *)table->Schema->FKeyHash.Find(table->Name, nameLength);
	}

	__device__ static void FKTriggerDelete(Context *ctx, Trigger *p)
	{
		if (p)
		{
			TriggerStep *step = p->StepList;
			Expr::Delete(ctx, step->Where);
			Expr::ListDelete(ctx, step->ExprList);
			Select::Delete(ctx, step->Select);
			Expr::Delete(ctx, p->When);
			_tagfree(ctx, p);
		}
	}

	__device__ void Parse::FKDropTable(SrcList *name, Table *table)
	{
		Context *ctx = Ctx;
		if ((ctx->Flags & Context::FLAG_ForeignKeys) && !IsVirtual(table) && !table->Select)
		{
			int skipId = 0;
			Vdbe *v = GetVdbe();

			_assert(v); // VDBE has already been allocated
			if (!FKReferences(table))
			{
				// Search for a deferred foreign key constraint for which this table is the child table. If one cannot be found, return without 
				// generating any VDBE code. If one can be found, then jump over the entire DELETE if there are no outstanding deferred constraints
				// when this statement is run.
				FKey *p;
				for (p = table->FKeys; p; p = p->NextFrom)
					if (p->IsDeferred) break;
				if (!p) return;
				skipId = v->MakeLabel();
				v->AddOp2(OP_FkIfZero, 1, skipId);
			}

			DisableTriggers = true;
			Delete::DeleteFrom(this, Expr::SrcListDup(ctx, name, 0), 0);
			DisableTriggers = false;

			// If the DELETE has generated immediate foreign key constraint violations, halt the VDBE and return an error at this point, before
			// any modifications to the schema are made. This is because statement transactions are not able to rollback schema changes.
			v->AddOp2(OP_FkIfZero, 0, v->CurrentAddr()+2);
			HaltConstraint(RC_CONSTRAINT_FOREIGNKEY, OE_Abort, "foreign key constraint failed", Vdbe::P4T_STATIC);

			if (skipId)
				v->ResolveLabel(skipId);
		}
	}

	__device__ void Parse::FKCheck(Table *table, int regOld, int regNew)
	{
		Context *ctx = Ctx; // Database handle
		bool isIgnoreErrors = DisableTriggers;

		// Exactly one of regOld and regNew should be non-zero.
		_assert((regOld == 0) != (regNew == 0));

		// If foreign-keys are disabled, this function is a no-op.
		if ((ctx->Flags & Context::FLAG_ForeignKeys) == 0) return;

		int db = Prepare::SchemaToIndex(ctx, table->Schema); // Index of database containing table
		const char *dbName = ctx->DBs[db].Name; // Name of database containing table

		// Loop through all the foreign key constraints for which table is the child table (the table that the foreign key definition is part of).
		FKey *fkey;
		for (fkey = table->FKeys; fkey; fkey = fkey->NextFrom)
		{
			bool isIgnore = false;
			// Find the parent table of this foreign key. Also find a unique index on the parent key columns in the parent table. If either of these 
			// schema items cannot be located, set an error in parse and return early.
			Table *to = (DisableTriggers ? FindTable(ctx, fkey->To, dbName) : LocateTable(false, fkey->To, dbName)); // Parent table of foreign key fkey
			Index *index = nullptr; // Index on key columns in to
			int *frees = nullptr;
			if (!to || FKLocateIndex(to, fkey, &index, &frees))
			{
				_assert(!isIgnoreErrors || (regOld != 0 && regNew == 0));
				if (!isIgnoreErrors || ctx->MallocFailed ) return;
				if (!to)
				{
					// If isIgnoreErrors is true, then a table is being dropped. In this se SQLite runs a "DELETE FROM xxx" on the table being dropped
					// before actually dropping it in order to check FK constraints. If the parent table of an FK constraint on the current table is
					// missing, behave as if it is empty. i.e. decrement the FK counter for each row of the current table with non-NULL keys.
					Vdbe *v = GetVdbe();
					int jumpId = v->CurrentAddr() + fkey->Cols.length + 1;
					for (int i = 0; i < fkey->Cols.length; i++)
					{
						int regId = fkey->Cols[i].From + regOld + 1;
						v->AddOp2(OP_IsNull, regId, jumpId);
					}
					v->AddOp2(OP_FkCounter, fkey->IsDeferred, -1);
				}
				continue;
			}
			_assert(fkey->Cols.length == 1 || (frees && index));

			int *cols;
			if (frees)
				cols = frees;
			else
			{
				int col = fkey->Cols[0].From;
				cols = &col;
			}
			for (int i = 0; i < fkey->Cols.length; i++)
			{
				if (cols[i] == table->PKey)
					cols[i] = -1;
#ifndef OMIT_AUTHORIZATION
				// Request permission to read the parent key columns. If the authorization callback returns SQLITE_IGNORE, behave as if any
				// values read from the parent table are NULL.
				if (ctx->Auth)
				{
					char *colName = to->Cols[index ? index->Columns[i] : to->PKey].Name;
					ARC rcauth = Auth::ReadColumn(this, to->Name, colName, db);
					isIgnore = (rcauth == ARC_IGNORE);
				}
#endif
			}

			// Take a shared-cache advisory read-lock on the parent table. Allocate a cursor to use to search the unique index on the parent key columns 
			// in the parent table.
			TableLock(db, to->Id, false, to->Name);
			Tabs++;

			if (regOld != 0) // A row is being removed from the child table. Search for the parent. If the parent does not exist, removing the child row resolves an outstanding foreign key constraint violation.
				FKLookupParent(this, db, to, index, fkey, cols, regOld, -1, isIgnore);
			if (regNew != 0) // A row is being added to the child table. If a parent row cannot be found, adding the child row has violated the FK constraint. 
				FKLookupParent(this, db, to, index, fkey, cols, regNew, +1, isIgnore);

			_tagfree(ctx, frees);
		}

		// Loop through all the foreign key constraints that refer to this table
		for (fkey = FKReferences(table); fkey; fkey = fkey->NextTo)
		{
			if (!fkey->IsDeferred && !Toplevel && !IsMultiWrite)
			{
				_assert(regOld == 0 && regNew != 0);
				// Inserting a single row into a parent table cannot cause an immediate foreign key violation. So do nothing in this case.
				continue;
			}

			Index *index = nullptr; // Foreign key index for fkey
			int *cols = 0;
			if (FKLocateIndex(table, fkey, &index, &cols))
			{
				if (!isIgnoreErrors || ctx->MallocFailed) return;
				continue;
			}
			_assert(cols || fkey->Cols.length == 1);

			// Create a SrcList structure containing a single table (the table the foreign key that refers to this table is attached to). This
			// is required for the sqlite3WhereXXX() interface.
			SrcList *src = SrcListAppend(ctx, nullptr, nullptr, nullptr);
			if (src)
			{
				SrcList::SrcListItem *item = src->Ids;
				item->Table = fkey->From;
				item->Name = fkey->From->Name;
				item->Table->Refs++;
				item->Cursor = Tabs++;

				if (regNew != 0)
					FKScanChildren(this, src, table, index, fkey, cols, regNew, -1);
				if (regOld != 0)
				{
					// If there is a RESTRICT action configured for the current operation on the parent table of this FK, then throw an exception 
					// immediately if the FK constraint is violated, even if this is a deferred trigger. That's what RESTRICT means. To defer checking
					// the constraint, the FK should specify NO ACTION (represented using OE_None). NO ACTION is the default.
					FKScanChildren(this, src, table, index, fkey, cols, regOld, 1);
				}
				item->Name = nullptr;
				SrcListDelete(ctx, src);
			}
			_tagfree(ctx, cols);
		}
	}

#define COLUMN_MASK(x) (((x)>31) ? 0xffffffff : ((uint32)1<<(x)))

	__device__ uint32 Parse::FKOldmask(Table *table)
	{
		uint32 mask = 0;
		if (Ctx->Flags & Context::FLAG_ForeignKeys)
		{
			FKey *p;
			int i;
			for (p = table->FKeys; p; p = p->NextFrom)
				for (i = 0; i < p->Cols.length; i++) mask |= COLUMN_MASK(p->Cols[i].From);
			for (p = FKReferences(table); p; p = p->NextTo)
			{
				Index *index = nullptr;
				FKLocateIndex(table, p, &index, nullptr);
				if (index)
					for (i = 0; i < index->Columns.length; i++) mask |= COLUMN_MASK(index->Columns[i]);
			}
		}
		return mask;
	}

	__device__ bool Parse::FKRequired(Table *table, int *changes, int chngRowid)
	{
		if (Ctx->Flags & Context::FLAG_ForeignKeys)
		{
			if (!changes) // A DELETE operation. Foreign key processing is required if the table in question is either the child or parent table for any foreign key constraint.
				return (FKReferences(table) || table->FKeys);
			else // This is an UPDATE. Foreign key processing is only required if operation modifies one or more child or parent key columns.
			{
				int i;
				FKey *p;
				// Check if any child key columns are being modified.
				for (p = table->FKeys; p; p = p->NextFrom)
				{
					for (i = 0; i < p->Cols.length; i++)
					{
						int childKeyId = p->Cols[i].From;
						if (changes[childKeyId] >= 0) return true;
						if (childKeyId == table->PKey && chngRowid) return true;
					}
				}

				// Check if any parent key columns are being modified.
				for (p = FKReferences(table); p; p = p->NextTo)
					for (i = 0; i < p->Cols.length; i++)
					{
						char *keyName = p->Cols[i].Col;
						for (int key = 0; key < table->Cols.length; key++)
						{
							Column *col = &table->Cols[key];
							if (keyName ? !_strcmp(col->Name, keyName) : (col->ColFlags & COLFLAG_PRIMKEY) != 0)
							{
								if (changes[key] >= 0) return true;
								if (key == table->PKey && chngRowid) return true;
							}
						}
					}
			}
		}
		return false;
	}

	__device__ static Trigger *FKActionTrigger(Parse *parse, Table *table, FKey *key, ExprList *changes)
	{
		Context *ctx = parse->Ctx; // Database handle
		int actionId = (changes != 0); // 1 for UPDATE, 0 for DELETE
		int action = key->Actions[actionId]; // One of OE_None, OE_Cascade etc.
		Trigger *trigger = key->Triggers[actionId]; // Trigger definition to return

		if (action != OE_None && !trigger)
		{
			Index *index = nullptr; // Parent key index for this FK
			int *cols = nullptr; // child table cols -> parent key cols
			if (parse->FKLocateIndex(table, key, &index, &cols)) return nullptr;
			_assert(cols || key->Cols.length == 1);

			Expr *where_ = nullptr; // WHERE clause of trigger step
			Expr *when = nullptr; // WHEN clause for the trigger
			ExprList *list = nullptr; // Changes list if ON UPDATE CASCADE
			for (int i = 0; i < key->Cols.length; i++)
			{
				Token oldToken = { "old", 3 };  // Literal "old" token
				Token newToken = { "new", 3 };  // Literal "new" token

				int fromColId = (cols ? cols[i] : key->Cols[0].From); // Idx of column in child table
				_assert(fromColId >= 0);
				Token fromCol; // Name of column in child table
				Token toCol; // Name of column in parent table
				toCol.data = (index ? table->Cols[index->Columns[i]].Name : "oid");
				fromCol.data = key->From->Cols[fromColId].Name;
				toCol.length = _strlen30(toCol.data);
				fromCol.length = _strlen30(fromCol.data);

				// Create the expression "OLD.zToCol = zFromCol". It is important that the "OLD.zToCol" term is on the LHS of the = operator, so
				// that the affinity and collation sequence associated with the parent table are used for the comparison.
				Expr *eq = Expr::PExpr_(parse, TK_EQ,
					Expr::PExpr_(parse, TK_DOT, 
					Expr::PExpr_(parse, TK_ID, 0, 0, &oldToken),
					Expr::PExpr_(parse, TK_ID, 0, 0, &toCol)
					, 0),
					Expr::PExpr_(parse, TK_ID, 0, 0, &fromCol)
					, 0); // tFromCol = OLD.tToCol
				where_ = Expr::And(ctx, where_, eq);

				// For ON UPDATE, construct the next term of the WHEN clause.
				// The final WHEN clause will be like this:
				//
				//    WHEN NOT(old.col1 IS new.col1 AND ... AND old.colN IS new.colN)
				if (changes)
				{
					eq = Expr::PExpr_(parse, TK_IS,
						Expr::PExpr_(parse, TK_DOT, 
						Expr::PExpr_(parse, TK_ID, 0, 0, &oldToken),
						Expr::PExpr_(parse, TK_ID, 0, 0, &toCol),
						0),
						Expr::PExpr_(parse, TK_DOT, 
						Expr::PExpr_(parse, TK_ID, 0, 0, &newToken),
						Expr::PExpr_(parse, TK_ID, 0, 0, &toCol),
						0),
						0);
					when = Expr::And(ctx, when, eq);
				}

				if (action != OE_Restrict && (action != OE_Cascade || changes))
				{
					Expr *newExpr;
					if (action == OE_Cascade)
						newExpr = Expr::PExpr_(parse, TK_DOT, 
						Expr::PExpr_(parse, TK_ID, 0, 0, &newToken),
						Expr::PExpr_(parse, TK_ID, 0, 0, &toCol)
						, 0);
					else if (action == OE_SetDflt)
					{
						Expr *dfltExpr = key->From->Cols[fromColId].Dflt;
						if (dfltExpr)
							newExpr = Expr::Dup(ctx, dfltExpr, 0);
						else
							newExpr = Expr::PExpr_(parse, TK_NULL, 0, 0, 0);
					}
					else
						newExpr = Expr::PExpr_(parse, TK_NULL, 0, 0, 0);
					list = Expr::ListAppend(parse, list, newExpr);
					Expr::ListSetName(parse, list, &fromCol, 0);
				}
			}
			_tagfree(ctx, cols);

			char const *fromName = key->From->Name; // Name of child table
			int fromNameLength = _strlen30(fromName); // Length in bytes of fromName

			Select *select = nullptr; // If RESTRICT, "SELECT RAISE(...)"
			if (action == OE_Restrict)
			{
				Token from;
				from.data = fromName;
				from.length = fromNameLength;
				Expr *raise = Expr::Expr_(ctx, TK_RAISE, "foreign key constraint failed");
				if (raise)
					raise->Aff = (AFF)OE_Abort; //:sky seems odd
				select = Select::New(parse, 
					Expr::ListAppend(parse, 0, raise),
					Parse::SrcListAppend(ctx, 0, &from, 0),
					where_,
					nullptr, nullptr, nullptr, (SF)0, nullptr, nullptr);
				where_ = nullptr;
			}

			// Disable lookaside memory allocation
			bool enableLookaside = ctx->Lookaside.Enabled; // Copy of ctx->lookaside.bEnabled
			ctx->Lookaside.Enabled = false;

			trigger = (Trigger *)_tagalloc2(ctx, 
				sizeof(Trigger) + // Trigger
				sizeof(TriggerStep) + // Single step in trigger program
				fromNameLength + 1 // Space for step->target.z
				, true);
			TriggerStep *step = nullptr; // First (only) step of trigger program
			if (trigger)
			{
				step = trigger->StepList = (TriggerStep *)&trigger[1];
				step->Target.data = (char *)&step[1];
				step->Target.length = fromNameLength;
				_memcpy(step->Target.data, fromName, fromNameLength);

				step->Where = Expr::Dup(ctx, where_, EXPRDUP_REDUCE);
				step->ExprList = Expr::ListDup(ctx, list, EXPRDUP_REDUCE);
				step->Select = Expr::SelectDup(ctx, select, EXPRDUP_REDUCE);
				if (when)
				{
					when = Expr::PExpr_(parse, TK_NOT, when, 0, 0);
					trigger->When = Expr::Dup(ctx, when, EXPRDUP_REDUCE);
				}
			}

			// Re-enable the lookaside buffer, if it was disabled earlier.
			ctx->Lookaside.Enabled = enableLookaside;

			Expr::Delete(ctx, where_);
			Expr::Delete(ctx, when);
			Expr::ListDelete(ctx, list);
			Select::Delete(ctx, select);
			if (ctx->MallocFailed)
			{
				FKTriggerDelete(ctx, trigger);
				return nullptr;
			}
			_assert(step);

			switch (action)
			{
			case OE_Restrict:
				step->OP = TK_SELECT; 
				break;
			case OE_Cascade: 
				if (!changes)
				{ 
					step->OP = TK_DELETE; 
					break; 
				}
			default:
				step->OP = TK_UPDATE;
			}
			step->Trig = trigger;
			trigger->Schema = table->Schema;
			trigger->TabSchema = table->Schema;
			key->Triggers[actionId] = trigger;
			trigger->OP = (changes ? TK_UPDATE : TK_DELETE);
		}

		return trigger;
	}

	__device__ void Parse::FKActions(Table *table, ExprList *changes, int regOld)
	{
		// If foreign-key support is enabled, iterate through all FKs that refer to table table. If there is an action associated with the FK 
		// for this operation (either update or delete), invoke the associated trigger sub-program.
		if (Ctx->Flags & Context::FLAG_ForeignKeys)
			for (FKey *fkey = FKReferences(table); fkey; fkey = fkey->NextTo)
			{
				Trigger *action = FKActionTrigger(this, table, fkey, changes);
				if (action)
					action->CodeRowTriggerDirect(this, table, regOld, OE_Abort, 0);
			}
	}

#endif

	__device__ void Parse::FKDelete(Context *ctx, Table *table)
	{
		_assert(!ctx || Btree::SchemaMutexHeld(ctx, 0, table->Schema));
		FKey *next; // Copy of pFKey->pNextFrom
		for (FKey *fkey = table->FKeys; fkey; fkey = next)
		{
			// Remove the FK from the fkeyHash hash table.
			if (!ctx || ctx->BytesFreed == 0)
			{
				if (fkey->PrevTo)
					fkey->PrevTo->NextTo = fkey->NextTo;
				else
				{
					void *p = (void *)fkey->NextTo;
					const char *z = (p ? fkey->NextTo->To : fkey->To);
					table->Schema->FKeyHash.Insert(z, _strlen30(z), p);
				}
				if (fkey->NextTo)
					fkey->NextTo->PrevTo = fkey->PrevTo;

			}

			// EV: R-30323-21917 Each foreign key constraint in SQLite is classified as either immediate or deferred.
			_assert(fkey->IsDeferred == false || fkey->IsDeferred == true);

			// Delete any triggers created to implement actions for this FK.
#ifndef OMIT_TRIGGER
			FKTriggerDelete(ctx, fkey->Triggers[0]);
			FKTriggerDelete(ctx, fkey->Triggers[1]);
#endif
			next = fkey->NextFrom;
			_tagfree(ctx, fkey);
		}
	}

}
#endif
#pragma endregion
