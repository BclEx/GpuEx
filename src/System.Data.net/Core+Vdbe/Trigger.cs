#region OMIT_TRIGGER
#if !OMIT_TRIGGER
using Core.Command;
using System;
using System.Diagnostics;
using System.Text;

namespace Core
{
    public partial class Trigger
    {
        public static void DeleteTriggerStep(Context ctx, ref TriggerStep triggerStep)
        {
            while (triggerStep != null)
            {
                TriggerStep tmp = triggerStep;
                triggerStep = triggerStep.Next;

                Expr.Delete(ctx, ref tmp.Where);
                Expr.ListDelete(ctx, ref tmp.ExprList);
                Select.Delete(ctx, ref tmp.Select);
                Parse.IdListDelete(ctx, ref tmp.IdList);
                
                C._tagfree(ctx, ref tmp);
                triggerStep = null; //: C#
            }
        }

        public static Trigger List(Parse parse, Table table)
        {
            Schema tmpSchema = parse.Ctx.DBs[1].Schema;
            Trigger list = null; // List of triggers to return
            if (parse.DisableTriggers)
                return null;
            if (tmpSchema != table.Schema)
            {
                Debug.Assert(Btree.SchemaMutexHeld(parse.Ctx, 0, tmpSchema));
                for (HashElem p = tmpSchema.TriggerHash.First; p != null; p = p.Next)
                {
                    Trigger trig = (Trigger)p.Data;
                    if (trig.TabSchema == table.Schema && string.Equals(trig.Table, table.Name, StringComparison.InvariantCultureIgnoreCase))
                    {
                        trig.Next = (list != null ? list : table.Triggers);
                        list = trig;
                    }
                }
            }
            return (list != null ? list : table.Triggers);
        }

        public static void BeginTrigger(Parse parse, Token name1, Token name2, TK trTm, TK op, IdList columns, SrcList tableName, Expr when, bool isTemp, int noErr)
        {
            Context ctx = parse.Ctx; // The database connection
            Debug.Assert(name1 != null);   // pName1.z might be NULL, but not pName1 itself
            Debug.Assert(name2 != null);
            Debug.Assert(op == TK.INSERT || op == TK.UPDATE || op == TK.DELETE);
            Debug.Assert(op > 0 && op < (TK)0xff);
            Trigger trigger = null; // The new trigger

            int db; // The database to store the trigger in
            Token name = null; // The unqualified db name
            if (isTemp)
            {
                // If TEMP was specified, then the trigger name may not be qualified.
                if (name2.length > 0)
                {
                    parse.ErrorMsg("temporary trigger may not have qualified name");
                    goto trigger_cleanup;
                }
                db = 1;
                name = name1;
            }
            else
            {
                // Figure out the db that the the trigger will be created in
                db = parse.TwoPartName(name1, name2, ref name);
                if (db < 0)
                    goto trigger_cleanup;
            }
            if (tableName == null || ctx.MallocFailed)
                goto trigger_cleanup;

            // A long-standing parser bug is that this syntax was allowed:
            //    CREATE TRIGGER attached.demo AFTER INSERT ON attached.tab ....
            //                                                 ^^^^^^^^
            // To maintain backwards compatibility, ignore the database name on pTableName if we are reparsing our of SQLITE_MASTER.
            if (ctx.Init.Busy && db != 1)
            {
                C._tagfree(ctx, ref tableName.Ids[0].Database);
                tableName.Ids[0].Database = null;
            }

            // If the trigger name was unqualified, and the table is a temp table, then set iDb to 1 to create the trigger in the temporary database.
            // If sqlite3SrcListLookup() returns 0, indicating the table does not exist, the error is caught by the block below.
            //? if (tableName == null) goto trigger_cleanup;
            Table table = Delete.SrcListLookup(parse, tableName); // Table that the trigger fires off of
            if (ctx.Init.Busy == null && name2.length == 0 && table != null && table.Schema == ctx.DBs[1].Schema)
                db = 1;

            // Ensure the table name matches database name and that the table exists
            if (ctx.MallocFailed) goto trigger_cleanup;
            Debug.Assert(tableName.Srcs == 1);
            DbFixer sFix = new DbFixer(); // State vector for the DB fixer
            if (sFix.FixInit(parse, db, "trigger", name) && sFix.FixSrcList(tableName))
                goto trigger_cleanup;
            table = Delete.SrcListLookup(parse, tableName);
            if (table == null)
            {
                // The table does not exist.
                if (ctx.Init.DB == 1)
                {
                    // Ticket #3810.
                    // Normally, whenever a table is dropped, all associated triggers are dropped too.  But if a TEMP trigger is created on a non-TEMP table
                    // and the table is dropped by a different database connection, the trigger is not visible to the database connection that does the
                    // drop so the trigger cannot be dropped.  This results in an "orphaned trigger" - a trigger whose associated table is missing.
                    ctx.Init.OrphanTrigger = true;
                }
                goto trigger_cleanup;
            }
            if (E.IsVirtual(table))
            {
                parse.ErrorMsg("cannot create triggers on virtual tables");
                goto trigger_cleanup;
            }

            // Check that the trigger name is not reserved and that no trigger of the specified name exists
            string nameAsString = Parse.NameFromToken(ctx, name);
            if (nameAsString == null || parse.CheckObjectName(nameAsString) != RC.OK)
                goto trigger_cleanup;
            Debug.Assert(Btree.SchemaMutexHeld(ctx, db, null));
            if (ctx.DBs[db].Schema.TriggerHash.Find(nameAsString, nameAsString.Length, (Trigger)null) != null)
            {
                if (noErr == 0)
                    parse.ErrorMsg("trigger %T already exists", name);
                else
                {
                    Debug.Assert(!ctx.Init.Busy);
                    parse.CodeVerifySchema(db);
                }
                goto trigger_cleanup;
            }

            // Do not create a trigger on a system table
            if (table.Name.StartsWith("sqlite_", StringComparison.InvariantCultureIgnoreCase))
            {
                parse.ErrorMsg("cannot create trigger on system table");
                parse.Errs++;
                goto trigger_cleanup;
            }

            // INSTEAD of triggers are only for views and views only support INSTEAD of triggers.
            if (table.Select != null && trTm != TK.INSTEAD)
            {
                parse.ErrorMsg("cannot create %s trigger on view: %S", (trTm == TK.BEFORE ? "BEFORE" : "AFTER"), tableName, 0);
                goto trigger_cleanup;
            }
            if (table.Select == null && trTm == TK.INSTEAD)
            {
                parse.ErrorMsg("cannot create INSTEAD OF trigger on table: %S", tableName, 0);
                goto trigger_cleanup;
            }

#if !OMIT_AUTHORIZATION
            {
                int tabDb = Prepare.SchemaToIndex(ctx, table.Schema); // Index of the database holding pTab
                AUTH code = AUTH.CREATE_TRIGGER;
                string dbName = ctx.DBs[tabDb].Name;
                string dbTrigName = (isTemp ? ctx.DBs[1].Name : dbName);
                if (tabDb == 1 || isTemp) code = AUTH.CREATE_TEMP_TRIGGER;
                if (Auth.Check(parse, code, nameAsString, table.Name, dbTrigName) != 0 || Auth.Check(parse, AUTH.INSERT, E.SCHEMA_TABLE(tabDb), 0, dbName))
                    goto trigger_cleanup;
            }
#endif

            // INSTEAD OF triggers can only appear on views and BEFORE triggers cannot appear on views.  So we might as well translate every
            // INSTEAD OF trigger into a BEFORE trigger.  It simplifies code elsewhere.
            if (trTm == TK.INSTEAD)
                trTm = TK.BEFORE;

            // Build the Trigger object
            trigger = new Trigger(); //: (Trigger *)_tagalloc(db, sizeof(Trigger), true);
            if (trigger == null) goto trigger_cleanup;
            trigger.Name = name;
            trigger.Table = tableName.Ids[0].Name; //: _tagstrdup(ctx, tableName->Ids[0].Name);
            trigger.Schema = ctx.DBs[db].Schema;
            trigger.TabSchema = table.Schema;
            trigger.OP = op;
            trigger.TRtm = (trTm == TK.BEFORE ? TRIGGER.BEFORE : TRIGGER.AFTER);
            trigger.When = Expr.Dup(db, when, E.EXPRDUP_REDUCE);
            trigger.Columns = Expr.IdListDup(ctx, columns);
            Debug.Assert(parse.NewTrigger == null);
            parse.NewTrigger = trigger;

        trigger_cleanup:
            C._tagfree(ctx, ref name);
            Expr.SrcListDelete(ctx, ref tableName);
            Expr.IdListDelete(ctx, ref columns);
            Expr.Delete(ctx, ref when);
            if (parse.NewTrigger == null)
                DeleteTrigger(ctx, ref trigger);
            else
                Debug.Assert(parse.NewTrigger == trigger);
        }

        public static void FinishTrigger(Parse parse, TriggerStep stepList, Token all)
        {
            Trigger trig = parse.NewTrigger; // Trigger being finished
            Context ctx = parse.Ctx; // The database
            Token nameToken = new Token(); // Trigger name for error reporting

            parse.NewTrigger = null;
            if (C._NEVER(parse.Errs != 0) || trig == null) goto triggerfinish_cleanup;
            string name = trig.Name; // Name of trigger
            int db = Prepare.SchemaToIndex(parse.Ctx, trig.Schema); // Database containing the trigger
            trig.StepList = stepList;
            while (stepList != null)
            {
                stepList.Trig = trig;
                stepList = stepList.Next;
            }
            nameToken.data = trig.Name;
            nameToken.length = (uint)nameToken.data.Length;
            DbFixer sFix = new DbFixer(); // Fixer object
            if (sFix.FixInit(parse, db, "trigger", nameToken) && sFix.FixTriggerStep(trig.StepList))
                goto triggerfinish_cleanup;

            // if we are not initializing, build the sqlite_master entry
            if (ctx.Init.Busy)
            {
                // Make an entry in the sqlite_master table
                Vdbe v = parse.GetVdbe();
                if (v == null) goto triggerfinish_cleanup;
                parse.BeginWriteOperation(0, db);
                string z = all.data.Substring(0, (int)all.length); //: _tagstrndup(ctx, (char *)all->data, all->length);
                parse.NestedParse("INSERT INTO %Q.%s VALUES('trigger',%Q,%Q,0,'CREATE TRIGGER %q')", ctx.DBs[db].Name, E.SCHEMA_TABLE(db), name, trig.Table, z);
                C._tagfree(ctx, ref z);
                parse.ChangeCookie(db);
                v.AddParseSchemaOp(db, C._mtagprintf(ctx, "type='trigger' AND name='%q'", name));
            }

            if (!ctx.Init.Busy)
            {
                Trigger link = trig;
                Debug.Assert(Btree.SchemaMutexHeld(ctx, db, null));
                trig = ctx.DBs[db].Schema.TriggerHash.Insert(name, name.Length, trig);
                if (trig != null)
                    ctx.MallocFailed = true;
                else if (link.Schema == link.TabSchema)
                {
                    int tableLength = link.Table.Length;
                    Table table = (Table)link.TabSchema.TableHash.Find(link.Table, tableLength, (Table)null);
                    Debug.Assert(table != null);
                    link.Next = table.Triggers;
                    table.Triggers = link;
                }
            }

        triggerfinish_cleanup:
            DeleteTrigger(ctx, ref trig);
            Debug.Assert(parse.NewTrigger == null);
            DeleteTriggerStep(ctx, ref stepList);
        }

        public static TriggerStep SelectStep(Context ctx, Select select)
        {
            TriggerStep triggerStep = new TriggerStep(); //: _tagalloc(ctx, sizeof(TriggerStep), true);
            if (triggerStep == null)
            {
                Select.Delete(ctx, ref select);
                return null;
            }
            triggerStep.OP = TK.SELECT;
            triggerStep.Select = select;
            triggerStep.Orconf = OE.Default;
            return triggerStep;
        }

        static TriggerStep TriggerStepAllocate(Context ctx, TK op, Token name)
        {
            TriggerStep triggerStep = new TriggerStep(); //: _tagalloc(ctx, sizeof(TriggerStep) + name->length, true);
            if (triggerStep != null)
            {
                string z = name.data;
                triggerStep.Target.data = z;
                triggerStep.Target.length = name.length;
                triggerStep.OP = op;
            }
            return triggerStep;
        }

        // OVERLOADS, so I don't need to rewrite parse.c
        public static TriggerStep TriggerInsertStep(Context ctx, Token tableName, IdList column, int null_4, int null_5, OE orconf) { return TriggerInsertStep(ctx, tableName, column, null, null, orconf); }
        public static TriggerStep TriggerInsertStep(Context ctx, Token tableName, IdList column, ExprList list, int null_5, OE orconf) { return TriggerInsertStep(ctx, tableName, column, list, null, orconf); }
        public static TriggerStep TriggerInsertStep(Context ctx, Token tableName, IdList column, int null_4, Select select, OE orconf) { return TriggerInsertStep(ctx, tableName, column, null, select, orconf); }
        public static TriggerStep TriggerInsertStep(Context ctx, Token tableName, IdList column, ExprList list, Select select, OE orconf)
        {
            Debug.Assert(list == null || select == null);
            Debug.Assert(list != null || select != null || ctx.MallocFailed);
            TriggerStep triggerStep = TriggerStepAllocate(ctx, TK.INSERT, tableName);
            if (triggerStep != null)
            {
                triggerStep.Select = Select.Dup(ctx, select, E.EXPRDUP_REDUCE);
                triggerStep.IdList = column;
                triggerStep.ExprList = Expr.ListDup(ctx, list, E.EXPRDUP_REDUCE);
                triggerStep.Orconf = orconf;
            }
            else
                Expr.IdListDelete(ctx, ref column);
            Expr.ListDelete(ctx, ref list);
            Select.Delete(ctx, ref select);
            return triggerStep;
        }

        public static TriggerStep TriggerUpdateStep(Context ctx, Token tableName, ExprList list, Expr where, OE orconf)
        {
            TriggerStep triggerStep = TriggerStepAllocate(ctx, TK.UPDATE, tableName);
            if (triggerStep != null)
            {
                triggerStep.ExprList = Expr.ListDup(ctx, list, E.EXPRDUP_REDUCE);
                triggerStep.Where = Expr.Dup(ctx, where, E.EXPRDUP_REDUCE);
                triggerStep.Orconf = orconf;
            }
            Expr.ListDelete(ctx, ref list);
            Expr.Delete(ctx, ref where);
            return triggerStep;
        }

        public static TriggerStep TriggerDeleteStep(Context ctx, Token tableName, Expr where_)
        {
            TriggerStep triggerStep = TriggerStepAllocate(ctx, TK.DELETE, tableName);
            if (triggerStep != null)
            {
                triggerStep.Where = Expr.Dup(ctx, where_, E.EXPRDUP_REDUCE);
                triggerStep.Orconf = OE.Default;
            }
            Expr.Delete(ctx, ref where_);
            return triggerStep;
        }

        public static void DeleteTrigger(Context ctx, ref Trigger trigger)
        {
            if (trigger == null) return;
            DeleteTriggerStep(ctx, ref trigger.StepList);
            C._tagfree(ctx, ref trigger.Name);
            C._tagfree(ctx, ref trigger.Table);
            Expr.Delete(ctx, ref trigger.When);
            Expr.IdListDelete(ctx, ref trigger.Columns);
            C._tagfree(ctx, ref trigger);
            trigger = null;
        }

        public static void DropTrigger(Parse parse, SrcList name, int noErr)
        {
            Context ctx = parse.Ctx;
            if (ctx.MallocFailed || parse.ReadSchema(parse) != RC.OK)
                goto drop_trigger_cleanup;

            Debug.Assert(name.Srcs == 1);
            string dbName = name.Ids[0].Database;
            string nameAsString = name.Ids[0].Name;
            int nameLength = nameAsString.Length;
            Debug.Assert(dbName != null || Btree.HoldsAllMutexes(ctx));
            Trigger trigger = null;
            for (int i = E.OMIT_TEMPDB; i < ctx.DBs.length; i++)
            {
                int j = (i < 2 ? i ^ 1 : i); // Search TEMP before MAIN
                if (dbName != null && !string.Equals(ctx.DBs[j].Name, dbName, StringComparison.InvariantCultureIgnoreCase)) continue;
                Debug.Assert(Btree.SchemaMutexHeld(ctx, j, null));
                trigger = ctx.DBs[j].Schema.TriggerHash.Find(nameAsString, nameLength, (Trigger)null);
                if (trigger != null) break;
            }
            if (trigger == null)
            {
                if (noErr == 0)
                    parse.ErrorMsg("no such trigger: %S", name, 0);
                else
                    parse.CodeVerifyNamedSchema(dbName);
                parse.CheckSchema = true;
                goto drop_trigger_cleanup;
            }
            DropTriggerPtr(parse, trigger);

        drop_trigger_cleanup:
            SrcListDelete(ctx, ref name);
        }

        static Table TableOfTrigger(Trigger trigger)
        {
            int tableLength = trigger.Table.Length;
            return trigger.TabSchema.TableHash.Find(trigger.Table, tableLength, (Table)null);
        }

        static readonly Vdbe.VdbeOpList[] _dropTrigger = new Vdbe.VdbeOpList[]  {
            new Vdbe.VdbeOpList( Core.OP.Rewind,     0, ADDR(9),  0),
            new Vdbe.VdbeOpList( Core.OP.String8,    0, 1,        0), // 1
            new Vdbe.VdbeOpList( Core.OP.Column,     0, 1,        2),
            new Vdbe.VdbeOpList( Core.OP.Ne,         2, ADDR(8),  1),
            new Vdbe.VdbeOpList( Core.OP.String8,    0, 1,        0), // 4: "trigger"
            new Vdbe.VdbeOpList( Core.OP.Column,     0, 0,        2),
            new Vdbe.VdbeOpList( Core.OP.Ne,         2, ADDR(8),  1),
            new Vdbe.VdbeOpList( Core.OP.Delete,     0, 0,        0),
            new Vdbe.VdbeOpList( Core.OP.Next,       0, ADDR(1),  0), // 8
        };
        public static void DropTriggerPtr(Parse parse, Trigger trigger)
        {
            Context ctx = parse.Ctx;
            int db = Prepare.SchemaToIndex(ctx, trigger.Schema);
            Debug.Assert(db >= 0 && db < ctx.DBs.length);
            Table table = TableOfTrigger(trigger);
            Debug.Assert(table != null);
            Debug.Assert(table.Schema == trigger.Schema || db == 1);
#if !OMIT_AUTHORIZATION
            {
                AUTH code = AUTH.DROP_TRIGGER;
                string dbName = ctx.DBs[db].Name;
                string tableName = E.SCHEMA_TABLE(db);
                if (db == 1) code = AUTH.DROP_TEMP_TRIGGER;
                if (Auth.Check(parse, code, trigger.Name, table.Name, dbName) || Auth.Check(parse, AUTH.DELETE, tableName, null, dbName))
                    return;
            }
#endif

            // Generate code to destroy the database record of the trigger.
            Debug.Assert(table != null);
            Vdbe v = parse.GetVdbe();
            if (v != null)
            {
                parse.BeginWriteOperation(0, db);
                parse.OpenMasterTable(db);
                int base_ = v.AddOpList(_dropTrigger.Length, _dropTrigger);
                v.ChangeP4(base_ + 1, trigger.Name, Vdbe.P4T.TRANSIENT);
                v.ChangeP4(base_ + 4, "trigger", Vdbe.P4T.STATIC);
                parse.ChangeCookie(db);
                v.AddOp2(Core.OP.Close, 0, 0);
                v.AddOp4(Core.OP.DropTrigger, db, 0, 0, trigger.Name, 0);
                if (parse.Mems < 3)
                    parse.Mems = 3;
            }
        }

        public static void UnlinkAndDeleteTrigger(Context ctx, int db, string name)
        {
            Debug.Assert(Btree.SchemaMutexHeld(ctx, db, null));
            Trigger trigger = ctx.DBs[db].Schema.TriggerHash.Insert(name, name.Length, (Trigger)null);
            if (C._ALWAYS(trigger != null))
            {
                if (trigger.Schema == trigger.TabSchema)
                {
                    Table table = TableOfTrigger(trigger);
                    //: Trigger** pp;
                    //: for (pp = &table->Triggers; *pp != trigger; pp = &((*pp)->Next)) ;
                    //: *pp = (*pp)->Next;
                    if (table.Triggers == trigger)
                        table.Triggers = trigger.Next;
                    else
                    {
                        Trigger cc = table.Triggers;
                        while (cc != null)
                        {
                            if (cc.Next == trigger)
                            {
                                cc.Next = cc.Next.Next;
                                break;
                            }
                            cc = cc.Next;
                        }
                        Debug.Assert(cc != null);
                    }
                }
                DeleteTrigger(ctx, ref trigger);
                ctx.Flags |= Context.FLAG.InternChanges;
            }
        }

        static bool CheckColumnOverlap(IdList idList, ExprList eList)
        {
            if (idList == null || C._NEVER(eList == null)) return true;
            for (int e = 0; e < eList.Exprs; e++)
                if (Expr.IdListIndex(idList, eList.Ids[e].Name) >= 0) return true;
            return false;
        }

        public static Trigger TriggersExist(Parse parse, Table table, TK op, ExprList changes, out TRIGGER maskOut)
        {
            TRIGGER mask = 0;
            Trigger list = null;
            if ((parse.Ctx.Flags & Context.FLAG.EnableTrigger) != 0)
                list = List(parse, table);
            Debug.Assert(list == null || !E.IsVirtual(table));
            for (Trigger p = list; p != null; p = p.Next)
                if (p.OP == op && CheckColumnOverlap(p.Columns, changes))
                    mask |= p.TRtm;
            maskOut = mask;
            return (mask != 0 ? list : null);
        }

        static SrcList TargetSrcList(Parse parse, TriggerStep step)
        {
            Context ctx = parse.Ctx;
            SrcList src = Parse.SrcListAppend(parse.Ctx, null, step.Target, null); // SrcList to be returned
            if (src != null)
            {
                Debug.Assert(src.Srcs > 0);
                Debug.Assert(src.Ids != null);
                int db = Prepare.SchemaToIndex(ctx, step.Trig.Schema); // Index of the database to use
                if (db == 0 || db >= 2)
                {
                    Debug.Assert(db < ctx.DBs.length);
                    src.Ids[src.Srcs - 1].Database = ctx.DBs[db].Name; //: _tagstrdup(ctx, ctx->DBs[db].Name);
                }
            }
            return src;
        }

        static int CodeTriggerProgram(Parse parse, TriggerStep stepList, OE orconf)
        {
            Vdbe v = parse.V;
            Context ctx = parse.Ctx;
            Debug.Assert(parse.TriggerTab != null && parse.Toplevel != null);
            Debug.Assert(stepList != null);
            Debug.Assert(v != null);
            for (TriggerStep step = stepList; step != null; step = step.Next)
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
                parse.Orconf = (orconf == OE.Default ? step.Orconf : orconf);

                switch (step.OP)
                {
                    case TK.UPDATE:
                        Update(parse,
                          TargetSrcList(parse, step),
                          Expr.ListDup(ctx, step.ExprList, 0),
                          Expr.Dup(ctx, step.Where, 0),
                          parse.Orconf);
                        break;
                    case TK.INSERT:
                        Insert(parse,
                          TargetSrcList(parse, step),
                          Expr.ListDup(ctx, step.ExprList, 0),
                          Select.Dup(ctx, step.Select, 0),
                          Expr.IdListDup(ctx, step.IdList),
                          parse.Orconf);
                        break;
                    case TK.DELETE:
                        DeleteFrom(parse,
                          TargetSrcList(parse, step),
                          Expr.Dup(ctx, step.Where, 0));
                        break;
                    default:
                        Debug.Assert(step.OP == TK.SELECT);
                        SelectDest sDest = new SelectDest();
                        Select select = Expr.SelectDup(ctx, step.Select, 0);
                        Select.DestInit(sDest, SRT.Discard, 0);
                        Select.Select_(parse, select, ref sDest);
                        Select.Delete(ctx, ref select);
                        break;
                }
                if (step.OP != TK.SELECT)
                    v.AddOp0(OP.ResetCount);
            }
            return 0;
        }

#if DEBUG
        static string OnErrorText(OE onError)
        {
            switch (onError)
            {
                case OE.Abort: return "abort";
                case OE.Rollback: return "rollback";
                case OE.Fail: return "fail";
                case OE.Replace: return "replace";
                case OE.Ignore: return "ignore";
                case OE.Default: return "default";
            }
            return "n/a";
        }
#endif

        static void TransferParseError(Parse to, Parse from)
        {
            Debug.Assert(from.ErrMsg == null || from.Errs != 0);
            Debug.Assert(to.ErrMsg== null || to.Errs != 0);
            if (to.Errs == 0)
            {
                to.ErrMsg = from.ErrMsg;
                to.Errs = from.Errs;
            }
            else
                C._tagfree(from.Ctx, ref from.ErrMsg);
        }

        static TriggerPrg CodeRowTrigger(Parse parse, Trigger trigger, Table table, OE orconf)
        {
            Parse top = E.Parse_Toplevel(parse);
            Context ctx = parse.Ctx; // Database handle

            Debug.Assert(trigger.Name == null || table == TableOfTrigger(trigger));
            Debug.Assert(top.V != null);

            // Allocate the TriggerPrg and SubProgram objects. To ensure that they are freed if an error occurs, link them into the Parse.pTriggerPrg 
            // list of the top-level Parse object sooner rather than later.
            TriggerPrg prg = new TriggerPrg(); // Value to return //: _tagalloc(ctx, sizeof(TriggerPrg), true);
            if (prg == null) return null;
            prg.Next = top.TriggerPrg;
            top.TriggerPrg = prg;
            Vdbe.SubProgram program; // Sub-vdbe for trigger program
            prg.Program = program = new Vdbe.SubProgram();// sqlite3DbMallocZero( db, sizeof( SubProgram ) );
            if (program == null) return null;
            top.V.LinkSubProgram(program);
            prg.Trigger = trigger;
            prg.Orconf = orconf;
            prg.Colmasks[0] = 0xffffffff;
            prg.Colmasks[1] = 0xffffffff;

            // Allocate and populate a new Parse context to use for coding the trigger sub-program.
            Parse subParse = new Parse(); // Parse context for sub-vdbe //: _stackalloc(ctx, sizeof(Parse), true);
            if (subParse == null) return null;
            NameContext sNC = new NameContext(); // Name context for sub-vdbe
            sNC.Parse = subParse;
            subParse.Ctx = ctx;
            subParse.TriggerTab = table;
            subParse.Toplevel = top;
            subParse.AuthContext = trigger.Name;
            subParse.TriggerOp = trigger.OP;
            subParse.QueryLoops = parse.QueryLoops;

            int endTrigger = 0; // Label to jump to if WHEN is false
            Vdbe v = subParse.GetVdbe(); // Temporary VM
            if (v != null)
            {
#if DEBUG
                v.Comment("Start: %s.%s (%s %s%s%s ON %s)",
                    trigger.Name, OnErrorText(orconf),
                    (trigger.TRtm == TRIGGER.BEFORE ? "BEFORE" : "AFTER"),
                    (trigger.OP == TK.UPDATE ? "UPDATE" : string.Empty),
                    (trigger.OP == TK.INSERT ? "INSERT" : string.Empty),
                    (trigger.OP == TK.DELETE ? "DELETE" : string.Empty),
                    table.Name);
#endif
#if !OMIT_TRACE
                v.ChangeP4(-1, C._mtagprintf(ctx, "-- TRIGGER %s", trigger.Name), Vdbe.P4T.DYNAMIC);
#endif

                // If one was specified, code the WHEN clause. If it evaluates to false (or NULL) the sub-vdbe is immediately halted by jumping to the 
                // OP_Halt inserted at the end of the program.
                if (trigger.When != null)
                {
                    Expr when = Expr.Dup(ctx, trigger.When, 0); // Duplicate of trigger WHEN expression
                    if (ResolveExprNames(sNC, ref when) == RC.OK && !ctx.MallocFailed)
                    {
                        endTrigger = v.MakeLabel();
                        subParse.IfFalse(when, endTrigger, RC_JUMPIFNULL);
                    }
                    Expr.Delete(ctx, ref when);
                }

                // Code the trigger program into the sub-vdbe.
                CodeTriggerProgram(subParse, trigger.StepList, orconf);

                // Insert an OP_Halt at the end of the sub-program.
                if (endTrigger != 0)
                    v.ResolveLabel(endTrigger);
                v.AddOp0(Core.OP.Halt);
#if DEBUG
                v.Comment("End: %s.%s", trigger.Name, OnErrorText(orconf));
#endif
                TransferParseError(parse, subParse);
                if (!ctx.MallocFailed)
                    program.Ops.data = v.TakeOpArray(ref program.Ops.length, ref top.MaxArgs);
                program.Mems = subParse.Mems;
                program.Csrs = subParse.Tabs;
                program.Token = trigger.GetHashCode();
                prg.Colmasks[0] = subParse.Oldmask;
                prg.Colmasks[1] = subParse.Newmask;
                Vdbe.Delete(v);
            }

            Debug.Assert(subParse.Ainc == null && subParse.ZombieTab == null);
            Debug.Assert(subParse.TriggerPrg == null && subParse.MaxArgs == 0);
            C._stackfree(ctx, ref subParse);

            return prg;
        }

        static TriggerPrg GetRowTrigger(Parse parse, Trigger trigger, Table table, OE orconf)
        {
            Parse root = E.Parse_Toplevel(parse);
            Debug.Assert(trigger.Name == null || table == TableOfTrigger(trigger));

            // It may be that this trigger has already been coded (or is in the process of being coded). If this is the case, then an entry with
            // a matching TriggerPrg.pTrigger field will be present somewhere in the Parse.pTriggerPrg list. Search for such an entry.
            TriggerPrg prg;
            for (prg = root.TriggerPrg; prg != null && (prg.Trigger != trigger || prg.Orconf != orconf); prg = prg.Next) ;
            // If an existing TriggerPrg could not be located, create a new one.
            if (prg == null)
                prg = CodeRowTrigger(parse, trigger, table, orconf);
            return prg;
        }

        public void CodeRowTriggerDirect(Parse parse, Table table, int reg, OE orconf, int ignoreJump)
        {
            Vdbe v = parse.GetVdbe(); // Main VM
            TriggerPrg prg = GetRowTrigger(parse, this, table, orconf);
            Debug.Assert(prg != null || parse.Errs != 0 || parse.Ctx.MallocFailed);

            // Code the OP_Program opcode in the parent VDBE. P4 of the OP_Program is a pointer to the sub-vdbe containing the trigger program.
            if (prg != null)
            {
                bool recursive = (Name != null && (parse.Ctx.Flags & Context.FLAG.RecTriggers) == 0);
                v.AddOp3(Core.OP.Program, reg, ignoreJump, ++parse.Mems);
                v.ChangeP4(-1, prg.Program, Vdbe.P4T.SUBPROGRAM);
#if DEBUG
                v.Comment("Call: %s.%s", (!string.IsNullOrEmpty(p.Name) ? p.Name : "fkey"), OnErrorText(orconf));
#endif
                // Set the P5 operand of the OP_Program instruction to non-zero if recursive invocation of this trigger program is disallowed. Recursive
                // invocation is disallowed if (a) the sub-program is really a trigger, not a foreign key action, and (b) the flag to enable recursive triggers is clear.
                v.ChangeP5((int)(recursive ? 1 : 0));
            }
        }

        public void CodeRowTrigger(Parse parse, TK op, ExprList changes, TRIGGER trtm, Table table, int reg, OE orconf, int ignoreJump)
        {
            Debug.Assert(op == TK.UPDATE || op == TK.INSERT || op == TK.DELETE);
            Debug.Assert(trtm == TRIGGER.BEFORE || trtm == TRIGGER.AFTER);
            Debug.Assert((op == TK.UPDATE) == (changes != null));

            for (Trigger p = this; p != null; p = p.Next)
            {
                // Sanity checking:  The schema for the trigger and for the table are always defined.  The trigger must be in the same schema as the table or else it must be a TEMP trigger.
                Debug.Assert(p.Schema != null);
                Debug.Assert(p.TabSchema != null);
                Debug.Assert(p.Schema == p.TabSchema || p.Schema == parse.Ctx.DBs[1].Schema);

                // Determine whether we should code this trigger
                if (p.OP == op && p.TRtm == trtm && CheckColumnOverlap(p.Columns, changes))
                    p.CodeRowTriggerDirect(parse, table, reg, orconf, ignoreJump);
            }
        }

        public uint TriggerColmask(Parse parse, ExprList changes, bool isNew, TRIGGER trtm, Table table, OE orconf)
        {
            TK op = (changes != null ? TK.UPDATE : TK.DELETE);
            int isNewId = (isNew ? 1 : 0);
            uint mask = 0;
            for (Trigger p = this; p != null; p = p.Next)
                if (p.OP == op && (trtm & p.TRtm) != 0 && CheckColumnOverlap(p.Columns, changes))
                {
                    TriggerPrg prg = GetRowTrigger(parse, p, table, orconf);
                    if (prg != null)
                        mask |= prg.Colmasks[isNewId];
                }
            return mask;
        }
    }
}
#endif
#endregion