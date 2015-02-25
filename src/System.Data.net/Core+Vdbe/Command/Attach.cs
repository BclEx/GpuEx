// attach.c
using System;
using System.Diagnostics;
using System.Text;

#region OMIT_ATTACH
#if !OMIT_ATTACH
namespace Core.Command
{
    public class Attach
    {
        static RC ResolveAttachExpr(NameContext name, Expr expr)
        {
            RC rc = RC.OK;
            if (expr != null)
            {
                if (expr.OP != TK.ID)
                {
                    rc = sqlite3ResolveExprNames(name, ref expr);
                    if (rc == RC.OK && !expr.IsConstant())
                    {
                        name.Parse.ErrorMsg("invalid name: \"%s\"", expr.u.Token);
                        return RC.ERROR;
                    }
                }
                else
                    expr.OP = TK.STRING;
            }
            return rc;
        }

        static void AttachFunc_(FuncContext fctx, int notUsed1, Mem[] argv)
        {
            Context ctx = sqlite3_context_db_handle(fctx);
            string file = Mem_Text(argv[0]);
            string name = Mem_Text(argv[1]);
            if (file == null) file = "";
            if (name == null) name = "";

            // Check for the following errors:
            //     * Too many attached databases,
            //     * Transaction currently open
            //     * Specified database name already being used.
            RC rc = 0;
            string errDyn = null;
            if (ctx.DBs.length >= ctx.Limits[(int)LIMIT.ATTACHED] + 2)
            {
                errDyn = SysEx.Mprintf(ctx, "too many attached databases - max %d", ctx.Limits[(int)LIMIT.ATTACHED]);
                goto attach_error;
            }
            if (!ctx.AutoCommit)
            {
                errDyn = SysEx.Mprintf(ctx, "cannot ATTACH database within transaction");
                goto attach_error;
            }
            for (int i = 0; i < ctx.DBs.length; i++)
            {
                string z = ctx.DBs[i].Name;
                Debug.Assert(z != null && name != null);
                if (z.Equals(name, StringComparison.OrdinalIgnoreCase))
                {
                    errDyn = SysEx.Mprintf(ctx, "database %s is already in use", name);
                    goto attach_error;
                }
            }

            // Allocate the new entry in the ctx->aDb[] array and initialize the schema hash tables.
            //: Realloc
            if (ctx.DBs.data.Length <= ctx.DBs.length)
                Array.Resize(ref ctx.DBs.data, ctx.DBs.length + 1);
            if (ctx.DBs.data == null) return;
            ctx.DBs[ctx.DBs.length] = new Context.DB();
            Context.DB newDB = ctx.DBs[ctx.DBs.length];
            //: _memset(newDB, 0, sizeof(*newDB));

            // Open the database file. If the btree is successfully opened, use it to obtain the database schema. At this point the schema may or may not be initialized.
            VSystem.OPEN flags = ctx.OpenFlags;
            VSystem vfs = null;
            string path = string.Empty;
            string err = string.Empty;
            rc = sqlite3ParseUri(ctx.Vfs.Name, file, ref flags, ref vfs, ref path, ref err);
            if (rc != RC.OK)
            {
                if (rc == RC.NOMEM) ctx.MallocFailed = true;
                sqlite3_result_error(fctx, err, -1);
                C._free(ref err);
                return;
            }
            Debug.Assert(vfs != null);
            flags |= VSystem.OPEN.MAIN_DB;
            rc = Btree.Open(vfs, path, ctx, ref newDB.Bt, 0, flags);
            C._free(ref path);

            ctx.DBs.length++;
            if (rc == RC.CONSTRAINT)
            {
                rc = RC.ERROR;
                errDyn = SysEx.Mprintf(ctx, "database is already attached");
            }
            else if (rc == RC.OK)
            {
                newDB.Schema = sqlite3SchemaGet(ctx, newDB.Bt);
                if (newDB.Schema == null)
                    rc = RC.NOMEM;
                else if (newDB.Schema.FileFormat != 0 && newDB.Schema.Encode != E.CTXENCODE(ctx))
                {
                    errDyn = SysEx.Mprintf(ctx, "attached databases must use the same text encoding as main database");
                    rc = RC.ERROR;
                }
                Pager pager = newDB.Bt.get_Pager();
                pager.LockingMode(ctx.DefaultLockMode);
                newDB.Bt.SecureDelete(ctx.DBs[0].Bt.SecureDelete(true));
            }
            newDB.SafetyLevel = 3;
            newDB.Name = name;
            if (rc == RC.OK && newDB.Name == null)
                rc = RC.NOMEM;

#if HAS_CODEC
            //extern int sqlite3CodecAttach(sqlite3*, int, const void*, int);
            //extern void sqlite3CodecGetKey(sqlite3*, int, void**, int*);
            if (rc == RC.OK)
            {
                int keyLength;
                string key;
                TYPE t = sqlite3_value_type(argv[2]);
                switch (t)
                {
                    case TYPE.INTEGER:
                    case TYPE.FLOAT:
                        errDyn = "Invalid key value";
                        rc = RC.ERROR;
                        break;

                    case TYPE.TEXT:
                    case TYPE.BLOB:
                        keyLength = Mem.Bytes(argv[2]);
                        key = sqlite3_value_blob(argv[2]).ToString();
                        rc = sqlite3CodecAttach(ctx, ctx.DBs.length - 1, key, keyLength);
                        break;

                    case TYPE.NULL:
                        // No key specified.  Use the key from the main database
                        sqlite3CodecGetKey(ctx, 0, out key, out keyLength);
                        if (keyLength > 0 || ctx.DBs[0].Bt.GetReserve() > 0)
                            rc = sqlite3CodecAttach(ctx, ctx.DBs.length - 1, key, keyLength);
                        break;
                }
            }
#endif
            // If the file was opened successfully, read the schema for the new database.
            // If this fails, or if opening the file failed, then close the file and remove the entry from the ctx->aDb[] array. i.e. put everything back the way we found it.
            if (rc == RC.OK)
            {
                Btree.EnterAll(ctx);
                rc = sqlite3Init(ctx, ref errDyn);
                Btree.LeaveAll(ctx);
            }
            if (rc != 0)
            {
                int db = ctx.DBs.length - 1;
                Debug.Assert(db >= 2);
                if (ctx.DBs[db].Bt != null)
                {
                    ctx.DBs[db].Bt.Close();
                    ctx.DBs[db].Bt = null;
                    ctx.DBs[db].Schema = null;
                }
                sqlite3ResetInternalSchema(ctx, -1);
                ctx.DBs.length = db;
                if (rc == RC.NOMEM || rc == RC.IOERR_NOMEM)
                {
                    ctx.MallocFailed = true;
                    C._tagfree(ctx, ref errDyn);
                    errDyn = SysEx.Mprintf(ctx, "out of memory");
                }
                else if (errDyn == null)
                    errDyn = SysEx.Mprintf(ctx, "unable to open database: %s", file);
                goto attach_error;
            }
            return;

        attach_error:
            // Return an error if we get here
            if (errDyn != null)
            {
                sqlite3_result_error(fctx, errDyn, -1);
                C._tagfree(ctx, ref errDyn);
            }
            if (rc != 0) sqlite3_result_error_code(fctx, rc);
        }

        static void DetachFunc_(FuncContext fctx, int notUsed1, Mem[] argv)
        {
            Context ctx = sqlite3_context_db_handle(fctx);
            string name = Mem_Text(argv[0]);
            if (name == null) name = string.Empty;
            StringBuilder err = new StringBuilder(200);

            int i;
            Context.DB db = null;
            for (i = 0; i < ctx.DBs.length; i++)
            {
                db = ctx.DBs[i];
                if (db.Bt == null) continue;
                if (string.Equals(db.Name, name, StringComparison.OrdinalIgnoreCase)) break;
            }
            if (i >= ctx.DBs.length)
            {
                err.AppendFormat("no such database: %s", name);
                goto detach_error;
            }
            if (i < 2)
            {
                err.AppendFormat("cannot detach database %s", name);
                goto detach_error;
            }
            if (!ctx.AutoCommit)
            {
                err.Append("cannot DETACH database within transaction");
                goto detach_error;
            }
            if (db.Bt.IsInReadTrans() || db.Bt.IsInBackup())
            {
                err.Append("database %s is locked", name);
                goto detach_error;
            }
            db.Bt.Close();
            db.Bt = null;
            db.Schema = null;
            sqlite3ResetInternalSchema(ctx, -1);
            return;

        detach_error:
            sqlite3_result_error(fctx, err.ToString(), -1);
        }

        static void CodeAttach(Parse parse, AUTH type, FuncDef func, Expr authArg, Expr filename, Expr dbName, Expr key)
        {
            Context ctx = parse.Ctx;

            NameContext sName;
            sName = new NameContext();
            sName.Parse = parse;

            if (ResolveAttachExpr(sName, filename) != RC.OK || ResolveAttachExpr(sName, dbName) != RC.OK || ResolveAttachExpr(sName, key) != RC.OK)
            {
                parse.Errs++;
                goto attach_end;
            }

#if !OMIT_AUTHORIZATION
            if (authArg != null)
            {
                string authArgToken = (authArg.OP == TK.STRING ? authArg.u.Token : null);
                ARC arc = Auth.Check(parse, type, authArgToken, null, null);
                if (arc != ARC.OK)
                    goto attach_end;
            }
#endif
            Vdbe v = parse.GetVdbe();
            int regArgs = Expr.GetTempRange(parse, 4);
            Expr.Code(parse, filename, regArgs);
            Expr.Code(parse, dbName, regArgs + 1);
            Expr.Code(parse, key, regArgs + 2);

            Debug.Assert(v != null || ctx.MallocFailed);
            if (v != null)
            {
                v.AddOp3(OP.Function, 0, regArgs + 3 - func.Args, regArgs + 3);
                Debug.Assert(func.Args == -1 || (func.Args & 0xff) == func.Args);
                v.ChangeP5((byte)(func.Args));
                v.ChangeP4(-1, func, Vdbe.P4T.FUNCDEF);

                // Code an OP_Expire. For an ATTACH statement, set P1 to true (expire this statement only). For DETACH, set it to false (expire all existing statements).
                v.AddOp1(OP.Expire, (type == AUTH.ATTACH ? 1 : 0));
            }

        attach_end:
            Expr.Delete(ctx, ref filename);
            Expr.Delete(ctx, ref dbName);
            Expr.Delete(ctx, ref key);
        }

        static FuncDef _detachFuncDef = new FuncDef
        {
            1,                  // nArg
            TEXTENCODE.UTF8,    // iPrefEnc
            (FUNC)0,            // flags
            null,               // pUserData
            null,               // pNext
            DetachFunc_,        // xFunc
            null,               // xStep
            null,               // xFinalize
            "sqlite_detach",    // zName
            null,               // pHash
            null                // pDestructor
        };
        static void Attach_Detach(Parse parse, Expr dbName) { CodeAttach(parse, AUTH.DETACH, _detachFuncDef, dbName, null, null, dbName); }

        static FuncDef _attachFuncDef = new FuncDef
        {
            3,                  // nArg
            TEXTENCODE.UTF8,    // iPrefEnc
            (FUNC)0,            // flags
            null,               // pUserData
            null,               // pNext
            AttachFunc_,        // xFunc
            null,               // xStep
            null,               // xFinalize
            "sqlite_attach",    // zName
            null,               // pHash
            null                // pDestructor
        };
        static void Attach_Attach(Parse parse, Expr p, Expr dbName, Expr key) { CodeAttach(parse, AUTH.ATTACH, _attachFuncDef, p, p, dbName, key); }
    }
}
#endif
#endregion

namespace Core
{
    public partial class DbFixer
    {
        public bool FixInit(Parse parse, int db, string type, Token name)
        {
            if (C._NEVER(db < 0) || db == 1) return false;
            Context ctx = parse.Ctx;
            Debug.Assert(ctx.DBs.length > db);
            Parse = parse;
            DB = ctx.DBs[db].Name;
            Schema = ctx.DBs[db].Schema;
            Type = type;
            Name = name;
            return true;
        }

        public bool FixSrcList(SrcList list)
        {
            if (NEVER(list == null)) return false;
            string db = DB;
            int i;
            SrcList.SrcListItem item;
            for (i = 0; i < list.Srcs; i++)
            {
                item = list.Ids[i];
                if (item.Database != null && !string.Equals(item.Database, db, StringComparison.OrdinalIgnoreCase))
                {
                    Parse.ErrorMsg("%s %T cannot reference objects in database %s", Type, Name, item.Database);
                    return true;
                }
                item.Database = null;
                item.Schema = Schema;
#if !OMIT_VIEW || !OMIT_TRIGGER
                if (FixSelect(item.Select) || FixExpr(item.On)) return true;
#endif
            }
            return false;
        }

#if !OMIT_VIEW || !OMIT_TRIGGER
        public bool FixSelect(Select select)
        {
            while (select != null)
            {
                if (FixExprList(select.EList) || FixSrcList(select.Src) || FixExpr(select.Where) || FixExpr(select.Having))
                    return true;
                select = select.Prior;
            }
            return false;
        }

        public bool FixExpr(Expr expr)
        {
            while (expr != null)
            {
                if (E.ExprHasAnyProperty(expr, EP.TokenOnly)) break;
                if (E.ExprHasProperty(expr, EP.xIsSelect))
                {
                    if (FixSelect(expr.x.Select)) return true;
                }
                else
                {
                    if (FixExprList(expr.x.List)) return true;
                }
                if (FixExpr(expr.Right))
                    return true;
                expr = expr.Left;
            }
            return false;
        }

        public bool FixExprList(ExprList list)
        {
            if (list == null) return false;
            int i;
            ExprList.ExprListItem item;
            for (i = 0; i < list.Exprs; i++)
            {
                item = list.Ids[i];
                if (FixExpr(item.Expr))
                    return true;
            }
            return false;
        }
#endif

#if !OMIT_TRIGGER
        public bool FixTriggerStep(TriggerStep step)
        {
            while (step != null)
            {
                if (FixSelect(step.Select) || FixExpr(step.Where) || FixExprList(step.ExprList))
                    return true;
                step = step.Next;
            }
            return false;
        }
#endif
    }
}
