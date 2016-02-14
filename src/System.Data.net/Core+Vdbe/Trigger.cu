// trigger.c
#ifndef OMIT_TRIGGER
#include "VdbeInt.cu.h"

namespace CORE_NAME
{
	__device__ void Trigger::DeleteTriggerStep(Context *ctx, TriggerStep *triggerStep)
	{
		while (triggerStep)
		{
			TriggerStep *tmp = triggerStep;
			triggerStep = triggerStep->Next;

			Expr::Delete(ctx, tmp->Where);
			Expr::ListDelete(ctx, tmp->ExprList);
			Select::Delete(ctx, tmp->Select);
			Parse::IdListDelete(ctx, tmp->IdList);

			_tagfree(ctx, tmp);
		}
	}

	__device__ Trigger *Trigger::List(Parse *parse, Core::Table *table)
	{
		Core::Schema *const tmpSchema = parse->Ctx->DBs[1].Schema;
		Trigger *list = nullptr; // List of triggers to return
		if (parse->DisableTriggers)
			return nullptr;
		if (tmpSchema != table->Schema)
		{
			_assert(Btree::SchemaMutexHeld(parse->Ctx, 0, tmpSchema));
			for (HashElem *p = tmpSchema->TriggerHash.First; p; p = p->Next)
			{
				Trigger *trig = (Trigger *)p->Data;
				if (trig->TabSchema == table->Schema && !_strcmp(trig->Table, table->Name))
				{
					trig->Next = (list ? list : table->Triggers);
					list = trig;
				}
			}
		}
		return (list ? list : table->Triggers);
	}

	__device__ void Trigger::BeginTrigger(Parse *parse, Token *name1, Token *name2, int trTm, int op, IdList *columns, SrcList *tableName, Expr *when, bool isTemp, int noErr)
	{
		Context *ctx = parse->Ctx; // The database connection
		_assert(name1); // pName1->z might be NULL, but not pName1 itself
		_assert(name2);
		_assert(op == TK_INSERT || op == TK_UPDATE || op == TK_DELETE);
		_assert(op > 0 && op<0xff);
		Trigger *trigger = nullptr;  // The new trigger

		int db; // The database to store the trigger in
		Token *name; // The unqualified db name
		if (isTemp)
		{
			// If TEMP was specified, then the trigger name may not be qualified.
			if (name2->length > 0)
			{
				parse->ErrorMsg("temporary trigger may not have qualified name");
				goto trigger_cleanup;
			}
			db = 1;
			name = name1;
		}
		else
		{
			// Figure out the db that the trigger will be created in
			db = parse->TwoPartName(name1, name2, &name);
			if (db < 0)
				goto trigger_cleanup;
		}
		if (!tableName || ctx->MallocFailed)
			goto trigger_cleanup;


		// A long-standing parser bug is that this syntax was allowed:
		//
		//    CREATE TRIGGER attached.demo AFTER INSERT ON attached.tab ....
		//                                                 ^^^^^^^^
		// To maintain backwards compatibility, ignore the database name on pTableName if we are reparsing our of SQLITE_MASTER.
		if (ctx->Init.Busy && db != 1)
		{
			_tagfree(ctx, tableName->Ids[0].Database);
			tableName->Ids[0].Database = nullptr;
		}

		// If the trigger name was unqualified, and the table is a temp table, then set iDb to 1 to create the trigger in the temporary database.
		// If sqlite3SrcListLookup() returns 0, indicating the table does not exist, the error is caught by the block below.
		Core::Table *table = Delete::SrcListLookup(parse, tableName); // Table that the trigger fires off of
		if (!ctx->Init.Busy && name2->length == 0 && table && table->Schema == ctx->DBs[1].Schema)
			db = 1;

		// Ensure the table name matches database name and that the table exists
		if (ctx->MallocFailed) goto trigger_cleanup;
		_assert(tableName->Srcs == 1);
		DbFixer sFix;  // State vector for the DB fixer
		if (sFix.FixInit(parse, db, "trigger", name) && sFix.FixSrcList(tableName))
			goto trigger_cleanup;
		table = Delete::SrcListLookup(parse, tableName);
		if (!table)
		{
			// The table does not exist.
			if (ctx->Init.DB == 1)
			{
				// Ticket #3810.
				// Normally, whenever a table is dropped, all associated triggers are dropped too.  But if a TEMP trigger is created on a non-TEMP table
				// and the table is dropped by a different database connection, the trigger is not visible to the database connection that does the
				// drop so the trigger cannot be dropped.  This results in an "orphaned trigger" - a trigger whose associated table is missing.
				ctx->Init.OrphanTrigger = true;
			}
			goto trigger_cleanup;
		}
		if (IsVirtual(table))
		{
			parse->ErrorMsg("cannot create triggers on virtual tables");
			goto trigger_cleanup;
		}

		// Check that the trigger name is not reserved and that no trigger of the specified name exists
		char *nameAsString = Parse::NameFromToken(ctx, name); // Name of the trigger
		if (!nameAsString || parse->CheckObjectName(nameAsString) != RC_OK)
			goto trigger_cleanup;
		_assert(Btree::SchemaMutexHeld(ctx, db, 0));
		if (ctx->DBs[db].Schema->TriggerHash.Find(nameAsString, _strlen(nameAsString)))
		{
			if (!noErr)
				parse->ErrorMsg("trigger %T already exists", name);
			else
			{
				_assert(!ctx->Init.Busy);
				parse->CodeVerifySchema(db);
			}
			goto trigger_cleanup;
		}

		// Do not create a trigger on a system table
		if (!_strncmp(table->Name, "sqlite_", 7))
		{
			parse->ErrorMsg("cannot create trigger on system table");
			parse->Errs++;
			goto trigger_cleanup;
		}

		// INSTEAD of triggers are only for views and views only support INSTEAD of triggers.
		if (table->Select && trTm != TK_INSTEAD)
		{
			parse->ErrorMsg("cannot create %s trigger on view: %S",  (trTm == TK_BEFORE ? "BEFORE" : "AFTER"), tableName, 0);
			goto trigger_cleanup;
		}
		if (!table->Select && trTm == TK_INSTEAD)
		{
			parse->ErrorMsg("cannot create INSTEAD OF trigger on table: %S", tableName, 0);
			goto trigger_cleanup;
		}

#ifndef OMIT_AUTHORIZATION
		{
			int tabDb = Schema::ToIndex(ctx, table->Schema); // Index of the database holding pTab
			AUTH code = AUTH_CREATE_TRIGGER;
			const char *dbName = ctx->DBs[tabDb].Name;
			const char *dbTrigName = (isTemp ? ctx->DBs[1].Name : dbName);
			if (tabDb == 1 || isTemp) code = AUTH_CREATE_TEMP_TRIGGER;
			if (Auth::Check(parse, code, nameAsString, table->Name, dbTrigName) || Auth::Check(parse, AUTH_INSERT, SCHEMA_TABLE(tabDb), 0, dbName))
				goto trigger_cleanup;
		}
#endif

		// INSTEAD OF triggers can only appear on views and BEFORE triggers cannot appear on views.  So we might as well translate every
		// INSTEAD OF trigger into a BEFORE trigger.  It simplifies code elsewhere.
		if (trTm == TK_INSTEAD)
			trTm = TK_BEFORE;

		// Build the Trigger object
		trigger = (Trigger *)_tagallocZero(ctx, sizeof(Trigger));
		if (!trigger) goto trigger_cleanup;
		trigger->Name = nameAsString; nameAsString = nullptr;
		trigger->Table = _tagstrdup(ctx, tableName->Ids[0].Name);
		trigger->Schema = ctx->DBs[db].Schema;
		trigger->TabSchema = table->Schema;
		trigger->OP = (uint8)op;
		trigger->TRtm = (trTm == TK_BEFORE ? TRIGGER_BEFORE : TRIGGER_AFTER);
		trigger->When = Expr::Dup(ctx, when, EXPRDUP_REDUCE);
		trigger->Columns = Expr::IdListDup(ctx, columns);
		_assert(parse->NewTrigger == 0);
		parse->NewTrigger = trigger;

trigger_cleanup:
		_tagfree(ctx, nameAsString);
		Parse::SrcListDelete(ctx, tableName);
		Parse::IdListDelete(ctx, columns);
		Expr::Delete(ctx, when);
		if (!parse->NewTrigger)
			DeleteTrigger(ctx, trigger);
		else
			_assert(parse->NewTrigger == trigger);
	}

	__device__ void Trigger::FinishTrigger(Parse *parse, TriggerStep *stepList, Token *all)
	{
		Trigger *trig = parse->NewTrigger; // Trigger being finished
		Context *ctx = parse->Ctx; // The database
		Token nameToken; // Trigger name for error reporting

		parse->NewTrigger = nullptr;
		if (_NEVER(parse->Errs) || !trig) goto triggerfinish_cleanup;
		char *name = trig->Name; // Name of trigger
		int db = Schema::ToIndex(ctx, trig->Schema); // Database containing the trigger
		trig->StepList = stepList;
		while (stepList)
		{
			stepList->Trig = trig;
			stepList = stepList->Next;
		}
		nameToken.data = trig->Name;
		nameToken.length = _strlen(nameToken.data);
		DbFixer sFix; // Fixer object
		if (sFix.FixInit(parse, db, "trigger", &nameToken) && sFix.FixTriggerStep(trig->StepList))
			goto triggerfinish_cleanup;

		// if we are not initializing, build the sqlite_master entry
		if (!ctx->Init.Busy)
		{
			// Make an entry in the sqlite_master table
			Vdbe *v = parse->GetVdbe();
			if (!v) goto triggerfinish_cleanup;
			parse->BeginWriteOperation(0, db);
			char *z = _tagstrndup(ctx, (char *)all->data, all->length);
			parse->NestedParse("INSERT INTO %Q.%s VALUES('trigger',%Q,%Q,0,'CREATE TRIGGER %q')", ctx->DBs[db].Name, SCHEMA_TABLE(db), name, trig->Table, z);
			_tagfree(ctx, z);
			parse->ChangeCookie(db);
			v->AddParseSchemaOp(db, _mtagprintf(ctx, "type='trigger' AND name='%q'", name));
		}

		if (ctx->Init.Busy)
		{
			Trigger *link = trig;
			_assert(Btree::SchemaMutexHeld(ctx, db, nullptr));
			trig = (Trigger *)ctx->DBs[db].Schema->TriggerHash.Insert(name, _strlen(name), trig);
			if (trig)
				ctx->MallocFailed = true;
			else if (link->Schema == link->TabSchema)
			{
				int tableLength = _strlen(link->Table);
				Core::Table *table = (Core::Table *)link->TabSchema->TableHash.Find(link->Table, tableLength);
				_assert(table);
				link->Next = table->Triggers;
				table->Triggers = link;
			}
		}

triggerfinish_cleanup:
		DeleteTrigger(ctx, trig);
		_assert(!parse->NewTrigger);
		DeleteTriggerStep(ctx, stepList);
	}

	__device__ TriggerStep *Trigger::SelectStep(Context *ctx, Select *select)
	{
		TriggerStep *triggerStep = (TriggerStep *)_tagallocZero(ctx, sizeof(TriggerStep));
		if (!triggerStep)
		{
			Select::Delete(ctx, select);
			return nullptr;
		}
		triggerStep->OP = TK_SELECT;
		triggerStep->Select = select;
		triggerStep->Orconf = OE_Default;
		return triggerStep;
	}

	__device__ static TriggerStep *TriggerStepAllocate(Context *ctx, TK op, Token *name)
	{
		TriggerStep *triggerStep = (TriggerStep *)_tagallocZero(ctx, sizeof(TriggerStep) + name->length);
		if (triggerStep)
		{
			char *z = (char *)&triggerStep[1];
			_memcpy(z, name->data, name->length);
			triggerStep->Target.data = z;
			triggerStep->Target.length = name->length;
			triggerStep->OP = op;
		}
		return triggerStep;
	}

	__device__ TriggerStep *Trigger::InsertStep(Context *ctx, Token *tableName, IdList *column, ExprList *list, Select *select, OE orconf)
	{
		_assert(list == 0 || select == 0);
		_assert(list != 0 || select != 0 || ctx->MallocFailed);
		TriggerStep *triggerStep = TriggerStepAllocate(ctx, TK_INSERT, tableName);
		if (triggerStep)
		{
			triggerStep->Select = Expr::SelectDup(ctx, select, EXPRDUP_REDUCE);
			triggerStep->IdList = column;
			triggerStep->ExprList = Expr::ListDup(ctx, list, EXPRDUP_REDUCE);
			triggerStep->Orconf = orconf;
		}
		else
			Parse::IdListDelete(ctx, column);
		Expr::ListDelete(ctx, list);
		Select::Delete(ctx, select);
		return triggerStep;
	}

	__device__ TriggerStep *Trigger::UpdateStep(Context *ctx, Token *tableName, ExprList *list, Expr *where, OE orconf)
	{
		TriggerStep *triggerStep = TriggerStepAllocate(ctx, TK_UPDATE, tableName);
		if (triggerStep)
		{
			triggerStep->ExprList = Expr::ListDup(ctx, list, EXPRDUP_REDUCE);
			triggerStep->Where = Expr::Dup(ctx, where, EXPRDUP_REDUCE);
			triggerStep->Orconf = orconf;
		}
		Expr::ListDelete(ctx, list);
		Expr::Delete(ctx, where);
		return triggerStep;
	}

	__device__ TriggerStep *Trigger::DeleteStep(Context *ctx, Token *tableName, Expr *where_)
	{
		TriggerStep *triggerStep = TriggerStepAllocate(ctx, TK_DELETE, tableName);
		if (triggerStep)
		{
			triggerStep->Where = Expr::Dup(ctx, where_, EXPRDUP_REDUCE);
			triggerStep->Orconf = OE_Default;
		}
		Expr::Delete(ctx, where_);
		return triggerStep;
	}

	__device__ void Trigger::DeleteTrigger(Context *ctx, Trigger *trigger)
	{
		if (!trigger) return;
		DeleteTriggerStep(ctx, trigger->StepList);
		_tagfree(ctx, trigger->Name);
		_tagfree(ctx, trigger->Table);
		Expr::Delete(ctx, trigger->When);
		Parse::IdListDelete(ctx, trigger->Columns);
		_tagfree(ctx, trigger);
	}

	__device__ void Trigger::DropTrigger(Parse *parse, SrcList *name, int noErr)
	{
		Context *ctx = parse->Ctx;
		if (ctx->MallocFailed || Prepare::ReadSchema(parse) != RC_OK)
			goto drop_trigger_cleanup;

		_assert(name->Srcs == 1);
		const char *dbName = name->Ids[0].Database;
		const char *nameAsString = name->Ids[0].Name;
		int nameLength = _strlen(nameAsString);
		_assert(dbName != 0 || Btree::HoldsAllMutexes(ctx));
		Trigger *trigger = nullptr;
		for (int i = E_OMIT_TEMPDB; i < ctx->DBs.length; i++)
		{
			int j = (i < 2 ? i^1 : i); // Search TEMP before MAIN
			if (dbName && _strcmp(ctx->DBs[j].Name, dbName)) continue;
			_assert(Btree::SchemaMutexHeld(ctx, j, nullptr));
			trigger = (Trigger *)ctx->DBs[j].Schema->TriggerHash.Find(nameAsString, nameLength);
			if (trigger) break;
		}
		if (!trigger)
		{
			if (!noErr)
				parse->ErrorMsg("no such trigger: %S", name, 0);
			else
				parse->CodeVerifyNamedSchema(dbName);
			parse->CheckSchema = true;
			goto drop_trigger_cleanup;
		}
		DropTriggerPtr(parse, trigger);

drop_trigger_cleanup:
		Parse::SrcListDelete(ctx, name);
	}

	__device__ static Table *TableOfTrigger(Trigger *trigger)
	{
		int tableLength = _strlen(trigger->Table);
		return (Table *)trigger->TabSchema->TableHash.Find(trigger->Table, tableLength);
	}

	__constant__ static const Vdbe::VdbeOpList _dropTrigger[] = {
		{ OP_Rewind,     0, ADDR(9),  0},
		{ OP_String8,    0, 1,        0}, // 1
		{ OP_Column,     0, 1,        2},
		{ OP_Ne,         2, ADDR(8),  1},
		{ OP_String8,    0, 1,        0}, // 4: "trigger"
		{ OP_Column,     0, 0,        2},
		{ OP_Ne,         2, ADDR(8),  1},
		{ OP_Delete,     0, 0,        0},
		{ OP_Next,       0, ADDR(1),  0}, // 8
	};
	__device__ void Trigger::DropTriggerPtr(Parse *parse, Trigger *trigger)
	{
		Context *ctx = parse->Ctx;
		int db = Schema::ToIndex(ctx, trigger->Schema);
		_assert(db >= 0 && db < ctx->DBs.length);
		Core::Table *table = TableOfTrigger(trigger);
		_assert(table);
		_assert(table->Schema == trigger->Schema || db == 1);
#ifndef OMIT_AUTHORIZATION
		{
			AUTH code = AUTH_DROP_TRIGGER;
			const char *dbName = ctx->DBs[db].Name;
			const char *tableName = SCHEMA_TABLE(db);
			if (db == 1) code = AUTH_DROP_TEMP_TRIGGER;
			if (Auth::Check(parse, code, trigger->Name, table->Name, dbName) || Auth::Check(parse, AUTH_DELETE, tableName, nullptr, dbName))
				return;
		}
#endif

		// Generate code to destroy the database record of the trigger.
		_assert(table);
		Vdbe *v = parse->GetVdbe();
		if (v)
		{
			parse->BeginWriteOperation(0, db);
			parse->OpenMasterTable(db);
			int base_ = v->AddOpList(_lengthof(_dropTrigger), _dropTrigger);
			v->ChangeP4(base_+1, trigger->Name, Vdbe::P4T_TRANSIENT);
			v->ChangeP4(base_+4, "trigger", Vdbe::P4T_STATIC);
			parse->ChangeCookie(db);
			v->AddOp2(OP_Close, 0, 0);
			v->AddOp4(OP_DropTrigger, db, 0, 0, trigger->Name, 0);
			if (parse->Mems < 3)
				parse->Mems = 3;
		}
	}

	__device__ void Trigger::UnlinkAndDeleteTrigger(Context *ctx, int db, const char *name)
	{
		_assert(Btree::SchemaMutexHeld(ctx, db, nullptr));
		Trigger *trigger = (Trigger *)ctx->DBs[db].Schema->TriggerHash.Insert(name, _strlen(name), 0);
		if (_ALWAYS(trigger))
		{
			if (trigger->Schema == trigger->TabSchema)
			{
				Core::Table *table = TableOfTrigger(trigger);
				Trigger **pp;
				for (pp = &table->Triggers; *pp != trigger; pp = &((*pp)->Next));
				*pp = (*pp)->Next;
			}
			DeleteTrigger(ctx, trigger);
			ctx->Flags |= Context::FLAG_InternChanges;
		}
	}

	__device__ static bool CheckColumnOverlap(IdList *idList, ExprList *eList)
	{
		if (!idList || _NEVER(!eList)) return true;
		for (int e = 0; e < eList->Exprs; e++)
			if (Parse::IdListIndex(idList, eList->Ids[e].Name) >= 0) return true;
		return false; 
	}

	__device__ Trigger *Trigger::TriggersExist(Parse *parse, Core::Table *table, int op, ExprList *changes, TRIGGER *maskOut)
	{
		TRIGGER mask = (TRIGGER)0;
		Trigger *list = nullptr;
		if ((parse->Ctx->Flags & Context::FLAG_EnableTrigger) != 0)
			list = List(parse, table);
		_assert(!list || !IsVirtual(table));
		for (Trigger *p = list; p; p = p->Next)
			if (p->OP == op && CheckColumnOverlap(p->Columns, changes))
				mask |= p->TRtm;
		if (maskOut)
			*maskOut = mask;
		return (mask ? list : nullptr);
	}

	__device__ static SrcList *TargetSrcList(Parse *parse, TriggerStep *step)
	{
		Context *ctx = parse->Ctx;
		SrcList *src = Parse::SrcListAppend(ctx, nullptr, &step->Target, nullptr); // SrcList to be returned
		if (src)
		{
			_assert(src->Srcs > 0);
			_assert(src->Ids != 0);
			int db = Schema::ToIndex(ctx, step->Trig->Schema); // Index of the database to use
			if (db == 0 || db >= 2)
			{
				_assert(db < ctx->DBs.length);
				src->Ids[src->Srcs-1].Database = _tagstrdup(ctx, ctx->DBs[db].Name);
			}
		}
		return src;
	}

	__device__ static int CodeTriggerProgram(Parse *parse, TriggerStep *stepList, OE orconf)
	{
		Vdbe *v = parse->V;
		Context *ctx = parse->Ctx;
		_assert(parse->TriggerTab && parse->Toplevel);
		_assert(stepList);
		_assert(v);
		for (TriggerStep *step = stepList; step; step = step->Next)
		{
			// Figure out the ON CONFLICT policy that will be used for this step of the trigger program. If the statement that caused this trigger
			// to fire had an explicit ON CONFLICT, then use it. Otherwise, use the ON CONFLICT policy that was specified as part of the trigger
			// step statement. Example:
			//
			//   CREATE TRIGGER AFTER INSERT ON t1 BEGIN;
			//     INSERT OR REPLACE INTO t2 VALUES(new.a, new.b);
			//   END;
			//
			//   INSERT INTO t1 ... ;            -- insert into t2 uses REPLACE policy
			//   INSERT OR IGNORE INTO t1 ... ;  -- insert into t2 uses IGNORE policy
			parse->Orconf = (orconf == OE_Default ? step->Orconf : (OE)orconf);

			// Clear the cookieGoto flag. When coding triggers, the cookieGoto variable is used as a flag to indicate to sqlite3ExprCodeConstants()
			// that it is not safe to refactor constants (this happens after the start of the first loop in the SQL statement is coded - at that 
			// point code may be conditionally executed, so it is no longer safe to initialize constant register values).
			_assert(parse->CookieGoto == 0 || parse->CookieGoto == -1);
			parse->CookieGoto = 0;

			switch (step->OP)
			{
			case TK_UPDATE:
				Update::Update_(parse, 
					TargetSrcList(parse, step),
					Expr::ListDup(ctx, step->ExprList, 0), 
					Expr::Dup(ctx, step->Where, 0), 
					parse->Orconf);
				break;

			case TK_INSERT:
				Insert::Insert_(parse, 
					TargetSrcList(parse, step),
					Expr::ListDup(ctx, step->ExprList, 0), 
					Expr::SelectDup(ctx, step->Select, 0), 
					Expr::IdListDup(ctx, step->IdList), 
					parse->Orconf);
				break;

			case TK_DELETE: {
				Delete::DeleteFrom(parse, 
					TargetSrcList(parse, step),
					Expr::Dup(ctx, step->Where, 0));
				break;
							}
			default: _assert(step->OP == TK_SELECT);
				SelectDest sDest;
				Select *select = Expr::SelectDup(ctx, step->Select, 0);
				Select::DestInit(&sDest, SRT_Discard, 0);
				Select::Select_(parse, select, &sDest);
				Select::Delete(ctx, select);
				break;
			} 
			if (step->OP != TK_SELECT)
				v->AddOp0(OP_ResetCount);
		}
		return 0;
	}

#ifdef _DEBUG
	__device__ static const char *OnErrorText(OE onError)
	{
		switch (onError)
		{
		case OE_Abort:    return "abort";
		case OE_Rollback: return "rollback";
		case OE_Fail:     return "fail";
		case OE_Replace:  return "replace";
		case OE_Ignore:   return "ignore";
		case OE_Default:  return "default";
		}
		return "n/a";
	}
#endif

	__device__ static void TransferParseError(Parse *to, Parse *from)
	{
		_assert(from->ErrMsg == nullptr || from->Errs);
		_assert(to->ErrMsg == nullptr || to->Errs);
		if (to->Errs == 0)
		{
			to->ErrMsg = from->ErrMsg;
			to->Errs = from->Errs;
		}
		else
			_tagfree(from->Ctx, from->ErrMsg);
	}

	__device__ static TriggerPrg *CodeRowTrigger(Parse *parse, Trigger *trigger, Table *table, OE orconf)
	{
		Parse *top = Parse_Toplevel(parse);
		Context *ctx = parse->Ctx; // Database handle

		_assert(trigger->Name == nullptr || table == TableOfTrigger(trigger));
		_assert(top->V);

		// Allocate the TriggerPrg and SubProgram objects. To ensure that they are freed if an error occurs, link them into the Parse.pTriggerPrg 
		// list of the top-level Parse object sooner rather than later.
		TriggerPrg *prg = (TriggerPrg *)_tagallocZero(ctx, sizeof(TriggerPrg)); // Value to return
		if (!prg) return nullptr;
		prg->Next = top->TriggerPrg;
		top->TriggerPrg = prg;
		Vdbe::SubProgram *program; // Sub-vdbe for trigger program
		prg->Program = program = (Vdbe::SubProgram *)_tagallocZero(ctx, sizeof(Vdbe::SubProgram));
		if (!program) return nullptr;
		top->V->LinkSubProgram(program);
		prg->Trigger = trigger;
		prg->Orconf = orconf;
		prg->Colmasks[0] = 0xffffffff;
		prg->Colmasks[1] = 0xffffffff;

		// Allocate and populate a new Parse context to use for coding the trigger sub-program.
		Parse *subParse = (Parse *)_stackallocZero(ctx, sizeof(Parse)); // Parse context for sub-vdbe
		if (!subParse) return nullptr;
		NameContext sNC; _memset(&sNC, 0, sizeof(sNC)); // Name context for sub-vdbe
		sNC.Parse = subParse;
		subParse->Ctx = ctx;
		subParse->TriggerTab = table;
		subParse->Toplevel = top;
		subParse->AuthContext = trigger->Name;
		subParse->TriggerOP = trigger->OP;
		subParse->QueryLoops = parse->QueryLoops;

		int endTrigger = 0; // Label to jump to if WHEN is false
		Vdbe *v = subParse->GetVdbe(); // Temporary VM
		if (v)
		{
			Vdbe_Comment(v, "Start: %s.%s (%s %s%s%s ON %s)", 
				trigger->Name, OnErrorText(orconf),
				(trigger->TRtm == TRIGGER_BEFORE ? "BEFORE" : "AFTER"),
				(trigger->OP == TK_UPDATE ? "UPDATE" : ""),
				(trigger->OP == TK_INSERT ? "INSERT" : ""),
				(trigger->OP == TK_DELETE ? "DELETE" : ""),
				table->Name);
#ifndef OMIT_TRACE
			v->ChangeP4(-1, _mtagprintf(ctx, "-- TRIGGER %s", trigger->Name), Vdbe::P4T_DYNAMIC);
#endif

			// If one was specified, code the WHEN clause. If it evaluates to false (or NULL) the sub-vdbe is immediately halted by jumping to the 
			// OP_Halt inserted at the end of the program.
			if (trigger->When)
			{
				Expr *when = Expr::Dup(ctx, trigger->When, 0); // Duplicate of trigger WHEN expression
				if (Walker::ResolveExprNames(&sNC, when) == RC_OK && !ctx->MallocFailed)
				{
					endTrigger = v->MakeLabel();
					when->IfFalse(subParse, endTrigger, AFF_BIT_JUMPIFNULL);
				}
				Expr::Delete(ctx, when);
			}

			// Code the trigger program into the sub-vdbe.
			CodeTriggerProgram(subParse, trigger->StepList, orconf);

			// Insert an OP_Halt at the end of the sub-program.
			if (endTrigger)
				v->ResolveLabel(endTrigger);
			v->AddOp0(OP_Halt);
			Vdbe_Comment(v, "End: %s.%s", trigger->Name, OnErrorText(orconf));

			TransferParseError(parse, subParse);
			if (!ctx->MallocFailed)
				program->Ops.data = v->TakeOpArray(&program->Ops.length, &top->MaxArgs);
			program->Mems = subParse->Mems;
			program->Csrs = subParse->Tabs;
			program->Onces = subParse->Onces;
			program->Token = (void *)trigger;
			prg->Colmasks[0] = subParse->Oldmask;
			prg->Colmasks[1] = subParse->Newmask;
			Vdbe::Delete(v);
		}

		_assert(!subParse->Ainc && !subParse->ZombieTab);
		_assert(!subParse->TriggerPrg && !subParse->MaxArgs);
		_stackfree(ctx, subParse);

		return prg;
	}

	__device__ static TriggerPrg *GetRowTrigger(Parse *parse, Trigger *trigger, Table *table, OE orconf)
	{
		Parse *root = Parse_Toplevel(parse);
		_assert(!trigger->Name || table == TableOfTrigger(trigger));

		// It may be that this trigger has already been coded (or is in the process of being coded). If this is the case, then an entry with
		// a matching TriggerPrg.pTrigger field will be present somewhere in the Parse.pTriggerPrg list. Search for such an entry.
		TriggerPrg *prg;
		for (prg = root->TriggerPrg; prg && (prg->Trigger != trigger || prg->Orconf != orconf); prg = prg->Next) ;
		// If an existing TriggerPrg could not be located, create a new one.
		if (!prg)
			prg = CodeRowTrigger(parse, trigger, table, orconf);
		return prg;
	}

	__device__ void Trigger::CodeRowTriggerDirect(Parse *parse, Core::Table *table, int reg, OE orconf, int ignoreJump)
	{
		Vdbe *v = parse->GetVdbe(); // DataEx VM
		TriggerPrg *prg = GetRowTrigger(parse, this, table, orconf);
		_assert(prg || parse->Errs || parse->Ctx->MallocFailed);

		// Code the OP_Program opcode in the parent VDBE. P4 of the OP_Program is a pointer to the sub-vdbe containing the trigger program.
		if (prg)
		{
			bool recursive = (Name && (parse->Ctx->Flags & Context::FLAG_RecTriggers) == 0);

			v->AddOp3(OP_Program, reg, ignoreJump, ++parse->Mems);
			v->ChangeP4(-1, (const char *)prg->Program, Vdbe::P4T_SUBPROGRAM);
#if _DEBUG
			Vdbe_Comment(v, "Call: %s.%s", (Name ? Name : "fkey"), OnErrorText(orconf));
#endif

			// Set the P5 operand of the OP_Program instruction to non-zero if recursive invocation of this trigger program is disallowed. Recursive
			// invocation is disallowed if (a) the sub-program is really a trigger, not a foreign key action, and (b) the flag to enable recursive triggers is clear.
			v->ChangeP5((uint8)recursive);
		}
	}

	__device__ void Trigger::CodeRowTrigger(Parse *parse, TK op, ExprList *changes, TRIGGER trtm, ::Table *table, int reg, OE orconf, int ignoreJump)
	{
		_assert(op == TK_UPDATE || op == TK_INSERT || op == TK_DELETE );
		_assert(trtm == TRIGGER_BEFORE || trtm == TRIGGER_AFTER );
		_assert((op == TK_UPDATE) == (changes != nullptr));

		for (Trigger *p = this; p; p = p->Next)
		{
			// Sanity checking:  The schema for the trigger and for the table are always defined.  The trigger must be in the same schema as the table or else it must be a TEMP trigger.
			_assert(p->Schema);
			_assert(p->TabSchema);
			_assert(p->Schema == p->TabSchema || p->Schema == parse->Ctx->DBs[1].Schema);

			// Determine whether we should code this trigger
			if (p->OP == op && p->TRtm == trtm && CheckColumnOverlap(p->Columns, changes))
				p->CodeRowTriggerDirect(parse, table, reg, orconf, ignoreJump);
		}
	}

	__device__ uint32 Trigger::Colmask(Parse *parse, ExprList *changes, bool isNew, TRIGGER trtm, ::Table *table, OE orconf)
	{
		const int op = (changes ? TK_UPDATE : TK_DELETE);
		int isNewId = (isNew ? 1 : 0);
		uint32 mask = 0;
		for (Trigger *p = this; p; p = p->Next)
			if (p->OP == op && (trtm & p->TRtm) && CheckColumnOverlap(p->Columns, changes))
			{
				TriggerPrg *prg = GetRowTrigger(parse, p, table, orconf);
				if (prg)
					mask |= prg->Colmasks[isNewId];
			}
			return mask;
	}

#endif
}