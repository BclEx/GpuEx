#region OMIT_VIRTUALTABLE
#if !OMIT_VIRTUALTABLE
using System;
using System.Diagnostics;
using System.Text;

namespace Core
{
    public class VTableContext
    {
        public VTable VTable;  // The virtual table being constructed
        public Table Table;      // The Table object to which the virtual table belongs
    }

    public partial class VTable
    {
        public static RC CreateModule(Context ctx, string name, ITableModule imodule, object aux, Action<object> destroy)
        {
            RC rc = RC.OK;
            MutexEx.Enter(ctx.Mutex);
            int nameLength = name.Length;
            if (ctx.Modules.Find(name, nameLength, (TableModule)null) != null)
                rc = SysEx.MISUSE_BKPT();
            else
            {
                TableModule module = new TableModule(); //: _tagalloc(ctx, sizeof(TableModule) + nameLength + 1)
                if (module != null)
                {
                    var nameCopy = name;
                    module.Name = nameCopy;
                    module.IModule = imodule;
                    module.Aux = aux;
                    module.Destroy = destroy;
                    TableModule del = (TableModule)ctx.Modules.Insert(nameCopy, nameLength, module);
                    Debug.Assert(del == null && del == module);
                    if (del != null)
                    {
                        ctx.MallocFailed = true;
                        C._tagfree(ctx, ref del);
                    }
                }
            }
            rc = SysEx.ApiExit(ctx, rc);
            if (rc != RC.OK && destroy != null) destroy(aux);
            MutexEx.Leave(ctx.Mutex);
            return rc;
        }

        public void Lock()
        {
            Refs++;
        }

        public static VTable GetVTable(Context ctx, Table table)
        {
            Debug.Assert(E.IsVirtual(table));
            VTable vtable;
            for (vtable = table.VTables; vtable != null && vtable.Ctx != ctx; vtable = vtable.Next) ;
            return vtable;
        }

        public void Unlock()
        {
            Context ctx = Ctx;
            Debug.Assert(ctx != null);
            Debug.Assert(Refs > 0);
            Debug.Assert(ctx.Magic == MAGIC.OPEN || ctx.Magic == MAGIC.ZOMBIE);
            Refs--;
            if (Refs == 0)
            {
                if (IVTable != null)
                    IVTable.IModule.Disconnect(IVTable);
                //C._tagfree(ctx, ref this);
            }
        }

        static VTable VTableDisconnectAll(Context ctx, Table table)
        {
            // Assert that the mutex (if any) associated with the BtShared database that contains table p is held by the caller. See header comments 
            // above function sqlite3VtabUnlockList() for an explanation of why this makes it safe to access the sqlite3.pDisconnect list of any
            // database connection that may have an entry in the p->pVTable list.
            Debug.Assert(ctx == null || Btree.SchemaMutexHeld(ctx, 0, table.Schema));
            VTable r = null;
            VTable vtable = table.VTables;
            table.VTables = null;
            while (vtable != null)
            {
                VTable next = vtable.Next;
                Context ctx2 = vtable.Ctx;
                Debug.Assert(ctx2 != null);
                if (ctx2 == ctx)
                {
                    r = vtable;
                    table.VTables = r;
                    r.Next = null;
                }
                else
                {
                    vtable.Next = ctx2.Disconnect;
                    ctx2.Disconnect = vtable;
                }
                vtable = next;
            }
            Debug.Assert(ctx == null || r != null);
            return r;
        }

        public static void Disconnect(Context ctx, Table table)
        {
            Debug.Assert(E.IsVirtual(table));
            Debug.Assert(Btree.HoldsAllMutexes(ctx));
            Debug.Assert(MutexEx.Held(ctx.Mutex));
            for (VTable pvtable = table.VTables; pvtable != null; pvtable = pvtable.Next)
                if (pvtable.Ctx == ctx)
                {
                    VTable vtable = pvtable;
                    vtable = vtable.Next;
                    vtable.Unlock();
                    break;
                }
        }

        public static void UnlockList(Context ctx)
        {
            Debug.Assert(Btree.HoldsAllMutexes(ctx));
            Debug.Assert(MutexEx.Held(ctx.Mutex));
            VTable vtable = ctx.Disconnect;
            ctx.Disconnect = null;
            if (vtable != null)
            {
                Vdbe.ExpirePreparedStatements(ctx);
                do
                {
                    VTable next = vtable.Next;
                    vtable.Unlock();
                    vtable = next;
                } while (vtable != null);
            }
        }

        public static void Clear(Context ctx, Table table)
        {
            if (ctx == null || ctx.BytesFreed == 0)
                VTableDisconnectAll(null, table);
            if (table.ModuleArgs.data != null)
            {
                for (int i = 0; i < table.ModuleArgs.length; i++)
                    C._tagfree(ctx, ref table.ModuleArgs.data[i]);
                C._tagfree(ctx, ref table.ModuleArgs.data);
            }
        }

        static void AddModuleArgument(Context ctx, Table table, string arg)
        {
            int i = table.ModuleArgs.length++;
            //: int bytes = sizeof(char*) * (1 + table->ModuleArgs.length);
            //: char** moduleArgs = (char**)_tagrealloc(ctx, table->ModuleArgs, bytes);
            if (table.ModuleArgs.data == null || table.ModuleArgs.data.Length < table.ModuleArgs.length)
                Array.Resize(ref table.ModuleArgs.data, 3 + table.ModuleArgs.length);
            if (table.ModuleArgs.data == null)
            {
                for (int j = 0; j < i; j++)
                    C._tagfree(ctx, ref table.ModuleArgs.data[j]);
                C._tagfree(ctx, ref arg);
                C._tagfree(ctx, ref table.ModuleArgs.data);
                table.ModuleArgs.length = 0;
            }
            else
            {
                table.ModuleArgs[i] = arg;
                //: table.ModuleArgs[i + 1] = null;
            }
            //: table.ModuleArgs.data = moduleArgs;
        }

        public static void BeginParse(Parse parse, Token name1, Token name2, Token moduleName, bool ifNotExists)
        {
            parse.StartTable(name1, name2, false, false, true, false);
            Table table = parse.NewTable; // The new virtual table
            if (table == null) return;
            Debug.Assert(table.Index == null);

            Context ctx = parse.Ctx; // Database connection
            int db = Prepare.SchemaToIndex(ctx, table.Schema); // The database the table is being created in
            Debug.Assert(db >= 0);

            table.TabFlags |= TF.Virtual;
            table.ModuleArgs.length = 0;
            AddModuleArgument(ctx, table, Parse.NameFromToken(ctx, moduleName));
            AddModuleArgument(ctx, table, null);
            AddModuleArgument(ctx, table, table.Name);
            parse.NameToken.length = (uint)parse.NameToken.data.Length; //: (int)(&moduleName[moduleName->length] - name1);

#if !OMIT_AUTHORIZATION
            // Creating a virtual table invokes the authorization callback twice. The first invocation, to obtain permission to INSERT a row into the
            // sqlite_master table, has already been made by sqlite3StartTable(). The second call, to obtain permission to create the table, is made now.
            if (table.ModuleArgs.data != null)
                Auth.Check(parse, AUTH.CREATE_VTABLE, table.Name, table.ModuleArgs[0], ctx.DBs[db].Name);
#endif
        }

        static void AddArgumentToVtab(Parse parse)
        {
            if (parse.Arg.data != null && C._ALWAYS(parse.NewTable != null))
            {
                string z = parse.Arg.data.Substring(0, (int)parse.Arg.length);
                //: uint length = parse.Arg.length;
                Context ctx = parse.Ctx;
                AddModuleArgument(ctx, parse.NewTable, z);
            }
        }

        public static void FinishParse(Parse parse, Token end)
        {
            Table table = parse.NewTable; // The table being constructed
            Context ctx = parse.Ctx; // The database connection
            if (table == null)
                return;
            AddArgumentToVtab(parse);
            parse.Arg.data = null;
            if (table.ModuleArgs.length < 1)
                return;

            // If the CREATE VIRTUAL TABLE statement is being entered for the first time (in other words if the virtual table is actually being
            // created now instead of just being read out of sqlite_master) then do additional initialization work and store the statement text
            // in the sqlite_master table.
            if (!ctx.Init.Busy)
            {
                // Compute the complete text of the CREATE VIRTUAL TABLE statement
                if (end != null)
                    parse.NameToken.length = (uint)parse.NameToken.data.Length; //: (int)(end->data - parse->NameToken) + end->length;
                string stmt = C._mtagprintf(ctx, "CREATE VIRTUAL TABLE %T", parse.NameToken); //.Z.Substring(0, parse.NameToken.length));

                // A slot for the record has already been allocated in the SQLITE_MASTER table.  We just need to update that slot with all
                // the information we've collected.  
                //
                // The VM register number pParse->regRowid holds the rowid of an entry in the sqlite_master table tht was created for this vtab
                // by sqlite3StartTable().
                int db = Prepare.SchemaToIndex(ctx, table.Schema);
                parse.NestedParse("UPDATE %Q.%s SET type='table', name=%Q, tbl_name=%Q, rootpage=0, sql=%Q WHERE rowid=#%d",
                  ctx.DBs[db].Name, E.SCHEMA_TABLE(db),
                  table.Name, table.Name,
                  stmt,
                  parse.RegRowid
                );
                C._tagfree(ctx, ref stmt);
                Vdbe v = parse.GetVdbe();
                parse.ChangeCookie(db);

                v.AddOp2(OP.Expire, 0, 0);
                string where_ = C._mtagprintf(ctx, "name='%q' AND type='table'", table.Name);
                v.AddParseSchemaOp(db, where_);
                v.AddOp4(OP.VCreate, db, 0, 0, table.Name, (Vdbe.P4T)table.Name.Length + 1);
            }

            // If we are rereading the sqlite_master table create the in-memory record of the table. The xConnect() method is not called until
            // the first time the virtual table is used in an SQL statement. This allows a schema that contains virtual tables to be loaded before
            // the required virtual table implementations are registered.
            else
            {
                Schema schema = table.Schema;
                string name = table.Name;
                int nameLength = name.Length;
                Debug.Assert(Btree.SchemaMutexHeld(ctx, 0, schema));
                Table oldTable = schema.TableHash.Insert(name, nameLength, table);
                if (oldTable != null)
                {
                    ctx.MallocFailed = true;
                    Debug.Assert(table == oldTable); // Malloc must have failed inside HashInsert()
                    return;
                }
                parse.NewTable = null;
            }
        }

        public static void ArgInit(Parse parse)
        {
            AddArgumentToVtab(parse);
            parse.Arg.data = null;
            parse.Arg.length = 0;
        }

        public static void ArgExtend(Parse parse, Token token)
        {
            Token arg = parse.Arg;
            if (arg.data == null)
            {
                arg.data = token.data;
                arg.length = token.length;
            }
            else
            {
                //: Debug.Assert(arg < token);
                arg.length += token.length + 1; //: (int)(&token[token->length] - arg);
            }
        }

        public delegate RC Construct_t(Context ctx, object aux, int argsLength, string[] args, out IVTable vtable, out string error);
        static RC VTableCallConstructor(Context ctx, Table table, TableModule module, Construct_t construct, ref string errorOut)
        {
            string moduleName = table.Name;
            if (moduleName == null)
                return RC.NOMEM;

            VTable vtable = new VTable();
            if (vtable == null)
            {
                C._tagfree(ctx, ref moduleName);
                return RC.NOMEM;
            }
            vtable.Ctx = ctx;
            vtable.Module = module;

            int db = sqlite3SchemaToIndex(ctx, table.Schema);
            table.ModuleArgs[1] = ctx.DBs[db].Name;

            // Invoke the virtual table constructor
            Debug.Assert(ctx.VTableCtx != null);
            Debug.Assert(construct != null);
            VTableContext sVtableCtx = new VTableContext();
            sVtableCtx.Table = table;
            sVtableCtx.VTable = vtable;
            VTableContext priorCtx = ctx.VTableCtx;
            ctx.VTableCtx = sVtableCtx;

            string[] args = table.ModuleArgs.data;
            int argsLength = table.ModuleArgs.length;
            string error = null;
            RC rc = construct(ctx, module.Aux, argsLength, args, out vtable.IVTable, out error);
            ctx.VTableCtx = null;
            if (rc == RC.NOMEM)
                ctx.MallocFailed = true;

            if (rc != RC.OK)
            {
                if (error == null)
                    errorOut = C._mtagprintf(ctx, "vtable constructor failed: %s", moduleName);
                else
                {
                    errorOut = error;
                    error = null; //: _free(error);
                }
                C._tagfree(ctx, ref vtable);
            }
            else if (C._ALWAYS(vtable.IVTable != null))
            {
                // Justification of ALWAYS():  A correct vtab constructor must allocate the sqlite3_vtab object if successful.
                vtable.IVTable.IModule = module.IModule;
                vtable.Refs = 1;
                if (sVtableCtx.Table != null)
                {
                    errorOut = C._mtagprintf(ctx, "vtable constructor did not declare schema: %s", table.Name);
                    vtable.Unlock();
                    rc = RC.ERROR;
                }
                else
                {
                    // If everything went according to plan, link the new VTable structure into the linked list headed by pTab->pVTable. Then loop through the 
                    // columns of the table to see if any of them contain the token "hidden". If so, set the Column COLFLAG_HIDDEN flag and remove the token from
                    // the type string.
                    vtable.Next = table.VTables;
                    table.VTables = vtable;
                    for (int col = 0; col < table.Cols.length; col++)
                    {
                        string type = table.Cols[col].Type;
                        if (type == null) continue;
                        int typeLength = type.Length;
                        int i = 0;
                        if (string.Compare("hidden", 0, type, 0, 6, StringComparison.OrdinalIgnoreCase) == 0 || (type.Length > 6 && type[6] != ' '))
                        {
                            for (i = 0; i < typeLength; i++)
                                if (string.Compare(" hidden", 0, type, i, 7, StringComparison.OrdinalIgnoreCase) == 0 && (i + 7 == type.Length || (type[i + 7] == '\0' || type[i + 7] == ' ')))
                                {
                                    i++;
                                    break;
                                }
                        }
                        if (i < typeLength)
                        {
                            StringBuilder type2 = new StringBuilder(type);
                            int del = 6 + (type2.Length > i + 6 ? 1 : 0);
                            int j;
                            for (j = i; (j + del) < typeLength; j++)
                                type2[j] = type2[j + del];
                            if (type2[i] == '\0' && i > 0)
                            {
                                Debug.Assert(type[i - 1] == ' ');
                                type2.Length = i; //: type[i - 1] = '\0';
                            }
                            table.Cols[col].ColFlags |= COLFLAG.HIDDEN;
                            table.Cols[col].Type = type.ToString().Substring(0, j);
                        }
                    }
                }
            }

            C._tagfree(ctx, ref moduleName);
            return rc;
        }

        public static RC CallConnect(Parse parse, Table table)
        {
            Debug.Assert(table != null);
            Context ctx = parse.Ctx;
            if ((table.TabFlags & TF.Virtual) == 0 || GetVTable(ctx, table) != null)
                return RC.OK;

            // Locate the required virtual table module
            string moduleName = table.ModuleArgs[0];
            TableModule module = (TableModule)ctx.Modules.Find(moduleName, moduleName.Length, (TableModule)null);
            if (module == null)
            {
                parse.ErrorMsg("no such module: %s", moduleName);
                return RC.ERROR;
            }

            string error = null;
            RC rc = VTableCallConstructor(ctx, table, module, module.IModule.Connect, ref error);
            if (rc != RC.OK)
                parse.ErrorMsg("%s", error);
            C._tagfree(ctx, ref error);
            return rc;
        }

        static RC GrowVTrans(Context ctx)
        {
            const int ARRAY_INCR = 5;
            // Grow the sqlite3.aVTrans array if required
            if ((ctx.VTrans.length % ARRAY_INCR) == 0)
            {
                //: int bytes = sizeof(IVTable*) * (ctx->VTrans.length + ARRAY_INCR);
                //: VTable** vtrans = (VTable**)_tagrealloc(ctx, (void*)ctx->VTrans, bytes);
                //: if (!vtrans)
                //:     return RC_NOMEM;
                //: _memset(&vtrans[ctx->VTrans.length], 0, sizeof(IVTable*) * ARRAY_INCR);
                //: ctx->VTrans = vtrans;
                Array.Resize(ref ctx.VTrans.data, ctx.VTrans.length + ARRAY_INCR);
            }
            return RC.OK;
        }

        static void AddToVTrans(Context ctx, VTable vtable)
        {
            // Add pVtab to the end of sqlite3.aVTrans
            ctx.VTrans[ctx.VTrans.length++] = vtable;
            vtable.Lock();
        }

        public static RC CallCreate(Context ctx, int db, string tableName, ref string errorOut)
        {

            Table table = Parse.FindTable(ctx, tableName, ctx.DBs[db].Name);
            Debug.Assert(table != null && (table.TabFlags & TF.Virtual) != 0 && table.VTables == null);

            // Locate the required virtual table module
            string moduleName = table.ModuleArgs[0];
            TableModule module = (TableModule)ctx.Modules.Find(moduleName, moduleName.Length, (TableModule)null);

            // If the module has been registered and includes a Create method, invoke it now. If the module has not been registered, return an 
            // error. Otherwise, do nothing.
            RC rc = RC.OK;
            if (module == null)
            {
                errorOut = C._mtagprintf(ctx, "no such module: %s", moduleName);
                rc = RC.ERROR;
            }
            else
                rc = VTableCallConstructor(ctx, table, module, module.IModule.Create, ref errorOut);

            // Justification of ALWAYS():  The xConstructor method is required to create a valid sqlite3_vtab if it returns SQLITE_OK.
            if (rc == RC.OK && C._ALWAYS(GetVTable(ctx, table) != null))
            {
                rc = GrowVTrans(ctx);
                if (rc == RC.OK)
                    AddToVTrans(ctx, GetVTable(ctx, table));
            }
            return rc;
        }

        public static RC DeclareVTable(Context ctx, string createTableName)
        {
            MutexEx.Enter(ctx.Mutex);
            Table table;
            if (ctx.VTableCtx == null || (table = ctx.VTableCtx.Table) == null)
            {
                sqlite3Error(ctx, RC.MISUSE, null);
                MutexEx.Leave(ctx.Mutex);
                return SysEx.MISUSE_BKPT();
            }
            Debug.Assert((table.TabFlags & TF.Virtual) != 0);

            RC rc = RC.OK;
            Parse parse = new Parse(); //: _stackalloc(ctx, sizeof(Parse));
            if (parse == null)
                rc = RC.NOMEM;
            else
            {
                parse.DeclareVTable = true;
                parse.Ctx = ctx;
                parse.QueryLoops = 1;

                string error = null;
                if (sqlite3RunParser(parse, createTableName, ref error) == RC.OK && parse.NewTable != null && !ctx.MallocFailed && parse.NewTable.Select == null && (parse.NewTable.TabFlags & TF.Virtual) == 0)
                {
                    if (table.Cols.data == null)
                    {
                        table.Cols.data = parse.NewTable.Cols.data;
                        table.Cols.length = parse.NewTable.Cols.length;
                        parse.NewTable.Cols.length = 0;
                        parse.NewTable.Cols.data = null;
                    }
                    ctx.VTableCtx.Table = null;
                }
                else
                {
                    sqlite3Error(ctx, RC.ERROR, (error != null ? "%s" : null), error);
                    C._tagfree(ctx, ref error);
                    rc = RC.ERROR;
                }
                parse.DeclareVTable = false;

                if (parse.V != null)
                    parse.V.Finalize();
                Parse.DeleteTable(ctx, ref parse.NewTable);
                parse = null; //: C._stackfree(ctx, parse);
            }

            Debug.Assert(((int)rc & 0xff) == (int)rc);
            rc = SysEx.ApiExit(ctx, rc);
            MutexEx.Leave(ctx.Mutex);
            return rc;
        }

        public static RC CallDestroy(Context ctx, int db, string tableName)
        {
            RC rc = RC.OK;
            Table table = Parse.FindTable(ctx, tableName, ctx.DBs[db].Name);
            if (C._ALWAYS(table != null && table.VTables != null))
            {
                VTable vtable = VTableDisconnectAll(ctx, table);
                Debug.Assert(rc == RC.OK);
                rc = vtable.Module.IModule.Destroy(vtable.IVTable);

                // Remove the sqlite3_vtab* from the aVTrans[] array, if applicable
                if (rc == RC.OK)
                {
                    Debug.Assert(table.VTables == vtable && vtable.Next == null);
                    vtable.IVTable = null;
                    table.VTables = null;
                    vtable.Unlock();
                }
            }
            return rc;
        }

        static void CallFinaliser(Context ctx, int offset)
        {
            if (ctx.VTrans.data != null)
            {
                for (int i = 0; i < ctx.VTrans.length; i++)
                {
                    VTable vtable = ctx.VTrans[i];
                    IVTable ivtable = vtable.IVTable;
                    if (ivtable != null)
                    {
                        Func<IVTable, int> x = null;
                        if (offset == 0)
                            x = ivtable.IModule.Rollback;
                        else if (offset == 1)
                            x = ivtable.IModule.Commit;
                        else
                            throw new InvalidOperationException();
                        if (x != null)
                            x(ivtable);
                    }
                    vtable.Savepoints = 0;
                    vtable.Unlock();
                }
                C._tagfree(ctx, ref ctx.VTrans.data);
                ctx.VTrans.length = 0;
                ctx.VTrans.data = null;
            }
        }

        public static RC Sync(Context ctx, ref string errorOut)
        {
            RC rc = RC.OK;
            VTable[] vtrans = ctx.VTrans.data;

            ctx.VTrans.data = null;
            for (int i = 0; rc == RC.OK && i < ctx.VTrans.length; i++)
            {
                Func<IVTable, int> x = null;
                IVTable ivtable = vtrans[i].IVTable;
                if (ivtable != null && (x = ivtable.IModule.Sync) != null)
                {
                    rc = (RC)x(ivtable);
                    C._tagfree(ctx, ref errorOut);
                    errorOut = ivtable.ErrMsg;
                    C._free(ref ivtable.ErrMsg);
                }
            }
            ctx.VTrans.data = vtrans;
            return rc;
        }

        public static RC Rollback(Context ctx)
        {
            CallFinaliser(ctx, 0); //: offsetof(IVTable, Rollback)
            return RC.OK;
        }

        public static RC Commit(Context ctx)
        {
            CallFinaliser(ctx, 1); //: offsetof(IVTable, Commit)
            return RC.OK;
        }

        public static RC Begin(Context ctx, VTable vtable)
        {
            // Special case: If ctx->aVTrans is NULL and ctx->nVTrans is greater than zero, then this function is being called from within a
            // virtual module xSync() callback. It is illegal to write to virtual module tables in this case, so return SQLITE_LOCKED.
            if (InSync(ctx))
                return RC.LOCKED;
            if (vtable == null)
                return RC.OK;
            RC rc = RC.OK;
            ITableModule imodule = vtable.IVTable.IModule;
            if (imodule.Begin != null)
            {
                // If pVtab is already in the aVTrans array, return early
                for (int i = 0; i < ctx.VTrans.length; i++)
                    if (ctx.VTrans[i] == vtable)
                        return RC.OK;
                // Invoke the xBegin method. If successful, add the vtab to the sqlite3.aVTrans[] array.
                rc = GrowVTrans(ctx);
                if (rc == RC.OK)
                {
                    rc = imodule.Begin(vtable.IVTable);
                    if (rc == RC.OK)
                        AddToVTrans(ctx, vtable);
                }
            }
            return rc;
        }

        public static RC Savepoint(Context ctx, IPager.SAVEPOINT op, int savepoint)
        {
            Debug.Assert(op == IPager.SAVEPOINT.RELEASE || op == IPager.SAVEPOINT.ROLLBACK || op == IPager.SAVEPOINT.BEGIN);
            Debug.Assert(savepoint >= 0);
            RC rc = RC.OK;
            if (ctx.VTrans.data != null)
                for (int i = 0; rc == RC.OK && i < ctx.VTrans.length; i++)
                {
                    VTable vtable = ctx.VTrans[i];
                    ITableModule itablemodule = vtable.Module.IModule;
                    if (vtable.IVTable != null && itablemodule.Version >= 2)
                    {
                        Func<VTable, int, int> method = null;
                        switch (op)
                        {
                            case IPager.SAVEPOINT.BEGIN:
                                method = itablemodule.Savepoint;
                                vtable.Savepoints = savepoint + 1;
                                break;
                            case IPager.SAVEPOINT.ROLLBACK:
                                method = itablemodule.RollbackTo;
                                break;
                            default:
                                method = itablemodule.Release;
                                break;
                        }
                        if (method != null && vtable.Savepoints > savepoint)
                            rc = (RC)method(vtable.IVTable, savepoint);
                    }
                }
            return rc;
        }

        public static FuncDef OverloadFunction(Context ctx, FuncDef def, int argsLength, Expr expr)
        {
            // Check to see the left operand is a column in a virtual table
            if (C._NEVER(expr == null)) return def;
            if (expr.OP != TK.COLUMN) return def;
            Table table = expr.Table;
            if (C._NEVER(table == null)) return def;
            if ((table.TabFlags & TF.Virtual) == 0) return def;
            IVTable ivtable = GetVTable(ctx, table).IVTable;
            Debug.Assert(ivtable != null);
            Debug.Assert(ivtable.IModule != null);
            ITableModule imodule = (ITableModule)ivtable.IModule;
            if (imodule.FindFunction == null) return def;

            // Call the xFindFunction method on the virtual table implementation to see if the implementation wants to overload this function 
            string lowerName = def.Name;
            RC rc = RC.OK;
            Action<FuncContext, int, Mem[]> func = null;
            object[] args = null;
            if (lowerName != null)
            {
                lowerName = lowerName.ToLowerInvariant();
                rc = imodule.FindFunction(ivtable, argsLength, lowerName, func, args);
                C._tagfree(ctx, ref lowerName);
            }
            if (rc == RC.OK)
                return def;

            // Create a new ephemeral function definition for the overloaded function
            FuncDef newFunc = new FuncDef();//: (FuncDef*)_tagalloc(ctx, sizeof(FuncDef) + _strlen30(def->Name) + 1, true);
            if (newFunc == null) return def;
            newFunc = def._memcpy();
            newFunc.Name = def.Name;
            newFunc.Func = func;
            newFunc.UserData = args;
            newFunc.Flags |= FUNC.EPHEM;
            return newFunc;
        }

        public static void MakeWritable(Parse parse, Table table)
        {
            Debug.Assert(E.IsVirtual(table));
            Parse toplevel = parse.Toplevel();
            for (int i = 0; i < toplevel.VTableLocks.length; i++)
                if (table == toplevel.VTableLocks[i]) return;
            int newSize = (toplevel.VTableLocks.data == null ? 1 : toplevel.VTableLocks.length + 1); //: (toplevel->VTableLocks.length + 1) * sizeof(toplevel->VTableLocks[0]);
            Array.Resize(ref toplevel.VTableLocks.data, newSize); //: Table vtablelocks = (Table **)_realloc(toplevel->VTableLocks, newSize);
            if (true) //vtablelocks != null)
            {
                //: toplevel.VTableLocks = vtablelocks;
                toplevel.VTableLocks[toplevel.VTableLocks.length++] = table;
            }
            else
                toplevel.Ctx.MallocFailed = true;
        }

        static CONFLICT[] _map = new[] {
            CONFLICT.ROLLBACK,
            CONFLICT.ABORT,
            CONFLICT.FAIL,
            CONFLICT.IGNORE, 
            CONFLICT.REPLACE
        };
        public static CONFLICT OnConflict(Context ctx)
        {
            Debug.Assert((int)OE.Rollback == 1 && (int)OE.Abort == 2 && (int)OE.Fail == 3);
            Debug.Assert((int)OE.Ignore == 4 && (int)OE.Replace == 5);
            Debug.Assert(ctx.VTableOnConflict >= 1 && ctx.VTableOnConflict <= 5);
            return _map[ctx.VTableOnConflict - 1];
        }

        public static RC Config(Context ctx, VTABLECONFIG op, object arg1)
        {
            RC rc = RC.OK;
            MutexEx.Enter(ctx.Mutex);
            switch (op)
            {
                case VTABLECONFIG.CONSTRAINT:
                    {
                        VTableContext p = ctx.VTableCtx;
                        if (p == null)
                            rc = SysEx.MISUSE_BKPT();
                        else
                        {
                            Debug.Assert(p.Table == null || (p.Table.TabFlags & TF.Virtual) != 0);
                            p.VTable.Constraint = (bool)arg1;
                        }
                        break;
                    }
                default:
                    rc = SysEx.MISUSE_BKPT();
                    break;
            }
            if (rc != RC.OK) Main.Error(ctx, rc, null);
            MutexEx.Leave(ctx.Mutex);
            return rc;
        }
    }
}
#endif
#endregion

