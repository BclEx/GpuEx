// attach.c
#include "..\Core+Vdbe.cu.h"

#pragma region OMIT_ATTACH
#ifndef OMIT_ATTACH
namespace Core { namespace Command
{
	__device__ static RC ResolveAttachExpr(NameContext *name, Expr *expr)
	{
		RC rc = RC_OK;
		if (expr)
		{
			if (expr->OP != TK_ID)
			{
				rc = (Walker::ResolveExprNames(name, expr) ? RC_OK : RC_ERROR);
				if (rc == RC_OK && !expr->IsConstant())
				{
					name->Parse->ErrorMsg("invalid name: \"%s\"", expr->u.Token);
					return RC_ERROR;
				}
			}
			else
				expr->OP = TK_STRING;
		}
		return rc;
	}

	__device__ static void AttachFunc(FuncContext *fctx, int notUsed1, Mem **argv)
	{
		Context *ctx = Vdbe::Context_Ctx(fctx);
		const char *file = (const char *)Vdbe::Value_Text(argv[0]);
		const char *name = (const char *)Vdbe::Value_Text(argv[1]);
		if (!file) file = "";
		if (!name) name = "";

		// Check for the following errors:
		//     * Too many attached databases,
		//     * Transaction currently open
		//     * Specified database name already being used.
		RC rc = (RC)0;
		char *errDyn = nullptr;
		if (ctx->DBs.length >= ctx->Limits[LIMIT_ATTACHED]+2)
		{
			errDyn = _mtagprintf(ctx, "too many attached databases - max %d", ctx->Limits[LIMIT_ATTACHED]);
			goto attach_error;
		}
		if (!ctx->AutoCommit)
		{
			errDyn = _mtagprintf(ctx, "cannot ATTACH database within transaction");
			goto attach_error;
		}
		for (int i = 0; i < ctx->DBs.length; i++)
		{
			char *z = ctx->DBs[i].Name;
			_assert(z && name);
			if (!_strcmp(z, name))
			{
				errDyn = _mtagprintf(ctx, "database %s is already in use", name);
				goto attach_error;
			}
		}

		// Allocate the new entry in the ctx->aDb[] array and initialize the schema hash tables.
		Context::DB *newDB;
		if (ctx->DBs.data == ctx->DBStatics)
		{
			newDB = (Context::DB *)_tagalloc(ctx, sizeof(ctx->DBs[0])*3);
			if (!newDB) return;
			_memcpy(newDB, ctx->DBs.data, sizeof(ctx->DBs[0])*2);
		}
		else
		{
			newDB = (Context::DB *)_tagrealloc(ctx, ctx->DBs.data, sizeof(ctx->DBs[0])*(ctx->DBs.length+1));
			if (!newDB) return;
		}
		ctx->DBs.data = newDB;
		newDB = &ctx->DBs[ctx->DBs.length];
		_memset(newDB, 0, sizeof(*newDB));

		// Open the database file. If the btree is successfully opened, use it to obtain the database schema. At this point the schema may or may not be initialized.
		VSystem::OPEN flags = ctx->OpenFlags;
		VSystem *vfs;
		char *path = nullptr;
		char *err = nullptr;
		rc = VSystem::ParseUri(ctx->Vfs->Name, file, &flags, &vfs, &path, &err);
		if (rc != RC_OK)
		{
			if (rc == RC_NOMEM) ctx->MallocFailed = true;
			Vdbe::Result_Error(fctx, err, -1);
			_free(err);
			return;
		}
		_assert(vfs);
		flags |= VSystem::OPEN_MAIN_DB;
		rc = Btree::Open(vfs, path, ctx, &newDB->Bt, (Btree::OPEN)0, flags);
		_free(path);
		ctx->DBs.length++;
		if (rc == RC_CONSTRAINT)
		{
			rc = RC_ERROR;
			errDyn = _mtagprintf(ctx, "database is already attached");
		}
		else if (rc == RC_OK)
		{
			newDB->Schema = Callback::SchemaGet(ctx, newDB->Bt);
			if (!newDB->Schema)
				rc = RC_NOMEM;
			else if (newDB->Schema->FileFormat && newDB->Schema->Encode != CTXENCODE(ctx))
			{
				errDyn = _mtagprintf(ctx, "attached databases must use the same text encoding as main database");
				rc = RC_ERROR;
			}
			Pager *pager = newDB->Bt->get_Pager();
			pager->LockingMode(ctx->DefaultLockMode);
			newDB->Bt->SecureDelete(ctx->DBs[0].Bt->SecureDelete(true));
		}
		newDB->SafetyLevel = 3;
		newDB->Name = _tagstrdup(ctx, name);
		if (rc == RC_OK && !newDB->Name)
			rc = RC_NOMEM;

#ifdef HAS_CODEC
		if (rc == RC_OK)
		{
			int keyLength;
			char *key;
			TYPE t = Vdbe::Value_Type(argv[2]);
			switch (t)
			{
			case TYPE_INTEGER:
			case TYPE_FLOAT:
				errDyn = _tagstrdup(ctx, "Invalid key value");
				rc = RC_ERROR;
				break;

			case TYPE_TEXT:
			case TYPE_BLOB:
				keyLength = Vdbe::Value_Bytes(argv[2]);
				key = (char *)Vdbe::Value_Blob(argv[2]);
				rc = Codec::Attach(ctx, ctx->DBs.length-1, key, keyLength);
				break;

			case TYPE_NULL:
				// No key specified.  Use the key from the main database
				Codec::GetKey(ctx, 0, (void**)&key, &keyLength);
				if (keyLength > 0 || ctx->DBs[0].Bt->GetReserve() > 0)
					rc = Codec::Attach(ctx, ctx->DBs.length-1, key, keyLength);
				break;
			}
		}
#endif

		// If the file was opened successfully, read the schema for the new database.
		// If this fails, or if opening the file failed, then close the file and remove the entry from the ctx->aDb[] array. i.e. put everything back the way we found it.
		if (rc == RC_OK)
		{
			Btree::EnterAll(ctx);
			rc = Prepare::Init(ctx, &errDyn);
			Btree::LeaveAll(ctx);
		}
		if (rc)
		{
			int db = ctx->DBs.length - 1;
			assert(db >= 2);
			if (ctx->DBs[db].Bt)
			{
				ctx->DBs[db].Bt->Close();
				ctx->DBs[db].Bt = nullptr;
				ctx->DBs[db].Schema = nullptr;
			}
			Parse::ResetAllSchemasOfConnection(ctx);
			ctx->DBs.length = db;
			if (rc == RC_NOMEM || rc == RC_IOERR_NOMEM)
			{
				ctx->MallocFailed = true;
				_tagfree(ctx, errDyn);
				errDyn = _mtagprintf(ctx, "out of memory");
			}
			else if (errDyn == nullptr)
				errDyn = _mtagprintf(ctx, "unable to open database: %s", file);
			goto attach_error;
		}
		return;

attach_error:
		// Return an error if we get here
		if (errDyn)
		{
			Vdbe::Result_Error(fctx, errDyn, -1);
			_tagfree(ctx, errDyn);
		}
		if (rc) Vdbe::Result_ErrorCode(fctx, rc);
	}

	__device__ static void DetachFunc(FuncContext *fctx, int NotUsed, Mem **argv)
	{
		Context *ctx = Vdbe::Context_Ctx(fctx);
		const char *name = (const char *)Vdbe::Value_Text(argv[0]);
		if (!name) name = "";
		char err[128];

		int i;
		Context::DB *db = nullptr;
		for (i = 0; i < ctx->DBs.length; i++)
		{
			db = &ctx->DBs[i];
			if (!db->Bt) continue;
			if (!_strcmp(db->Name, name)) break;
		}
		if (i >= ctx->DBs.length)
		{
			__snprintf(err, sizeof(err), "no such database: %s", name);
			goto detach_error;
		}
		if (i < 2)
		{
			__snprintf(err, sizeof(err), "cannot detach database %s", name);
			goto detach_error;
		}
		if (!ctx->AutoCommit)
		{
			__snprintf(err, sizeof(err), "cannot DETACH database within transaction");
			goto detach_error;
		}
		if (db->Bt->IsInReadTrans() || db->Bt->IsInBackup())
		{
			__snprintf(err, sizeof(err), "database %s is locked", name);
			goto detach_error;
		}
		db->Bt->Close();
		db->Bt = nullptr;
		db->Schema = nullptr;
		Parse::ResetAllSchemasOfConnection(ctx);
		return;

detach_error:
		Vdbe::Result_Error(fctx, err, -1);
	}

	__device__ static void CodeAttach(Parse *parse, AUTH type, FuncDef const *func, Expr *authArg, Expr *filename, Expr *dbName, Expr *key)
	{
		Context *ctx = parse->Ctx;

		NameContext sName;
		_memset(&sName, 0, sizeof(NameContext));
		sName.Parse = parse;

		if (ResolveAttachExpr(&sName, filename) != RC_OK || ResolveAttachExpr(&sName, dbName) != RC_OK || ResolveAttachExpr(&sName, key) != RC_OK)
		{
			parse->Errs++;
			goto attach_end;
		}

#ifndef OMIT_AUTHORIZATION
		if (authArg)
		{
			char *authArgToken = (authArg->OP == TK_STRING ? authArg->u.Token : nullptr);
			ARC arc = Auth::Check(parse, type, authArgToken, nullptr, nullptr);
			if (arc != ARC_OK)
				goto attach_end;
		}
#endif
		Vdbe *v = parse->GetVdbe();
		int regArgs = Expr::GetTempRange(parse, 4);
		Expr::Code(parse, filename, regArgs);
		Expr::Code(parse, dbName, regArgs+1);
		Expr::Code(parse, key, regArgs+2);

		_assert(v || ctx->MallocFailed);
		if (v)
		{
			v->AddOp3( OP_Function, 0, regArgs+3-func->Args, regArgs+3);
			_assert(func->Args == -1 || (func->Args & 0xff) == func->Args);
			v->ChangeP5((uint8)(func->Args));
			v->ChangeP4(-1, (char *)func, Vdbe::P4T_FUNCDEF);

			// Code an OP_Expire. For an ATTACH statement, set P1 to true (expire this statement only). For DETACH, set it to false (expire all existing statements).
			v->AddOp1(OP_Expire, (type == AUTH_ATTACH));
		}

attach_end:
		Expr::Delete(ctx, filename);
		Expr::Delete(ctx, dbName);
		Expr::Delete(ctx, key);
	}

	__device__ static const FuncDef _detachFuncDef = {
		1,					// nArg
		TEXTENCODE_UTF8,    // iPrefEnc
		(FUNC)0,			// flags
		0,					// pUserData
		0,					// pNext
		DetachFunc,			// xFunc
		0,					// xStep
		0,                // xFinalize
		"sqlite_detach",  // zName
		0,                // pHash
		0                 // pDestructor
	};
	__device__ void Attach::Detach(Parse *parse, Expr *dbName)
	{
		CodeAttach(parse, AUTH_DETACH, &_detachFuncDef, dbName, 0, 0, dbName);
	}

	__device__ static const FuncDef _attachFuncDef = {
		3,					// nArg
		TEXTENCODE_UTF8,	// iPrefEnc
		(FUNC)0,			// flags
		0,					// pUserData
		0,					// pNext
		AttachFunc,			// xFunc
		0,					// xStep
		0,					// xFinalize
		"sqlite_attach",	// zName
		0,					// pHash
		0					// pDestructor
	};
	__device__ void Attach::Attach_(Parse *parse, Expr *p, Expr *dbName, Expr *key)
	{
		CodeAttach(parse, AUTH_ATTACH, &_attachFuncDef, p, p, dbName, key);
	}
}}
#endif
#pragma endregion

namespace Core {

	__device__ bool DbFixer::FixInit(Core::Parse *parse, int db, const char *typeName, const Token *name)
	{
		if (_NEVER(db < 0) || db == 1) return false;
		Context *ctx = parse->Ctx;
		_assert(ctx->DBs.length > db);
		Parse = parse;
		DB = ctx->DBs[db].Name;
		Schema = ctx->DBs[db].Schema;
		Type = typeName;
		Name = name;
		return true;
	}

	__device__ bool DbFixer::FixSrcList(SrcList *list)
	{
		if (_NEVER(list == nullptr)) return false;
		const char *db = DB;
		int i;
		SrcList::SrcListItem *item;
		for (i = 0, item = list->Ids; i < list->Srcs; i++, item++)
		{
			if (item->Database && _strcmp(item->Database, db))
			{
				Parse->ErrorMsg("%s %T cannot reference objects in database %s", Type, Name, item->Database);
				return true;
			}
			_tagfree(Parse->Ctx, item->Database);
			item->Database = nullptr;
			item->Schema = Schema;
#if !defined(OMIT_VIEW) || !defined(OMIT_TRIGGER)
			if (FixSelect(item->Select) || FixExpr(item->On)) return true;
#endif
		}
		return false;
	}

#if !defined(OMIT_VIEW) || !defined(OMIT_TRIGGER)
	__device__ bool DbFixer::FixSelect(Select *select)
	{
		while (select)
		{
			if (FixExprList(select->EList) || FixSrcList(select->Src) || FixExpr(select->Where) || FixExpr(select->Having))
				return true;
			select = select->Prior;
		}
		return false;
	}

	__device__ bool DbFixer::FixExpr(Expr *expr)
	{
		while (expr)
		{
			if (ExprHasAnyProperty(expr, EP_TokenOnly)) break;
			if (ExprHasProperty(expr, EP_xIsSelect))
			{
				if (FixSelect(expr->x.Select)) return true;
			}
			else
			{
				if (FixExprList(expr->x.List)) return true;
			}
			if (FixExpr(expr->Right))
				return true;
			expr = expr->Left;
		}
		return false;
	}

	__device__ bool DbFixer::FixExprList(ExprList *list)
	{
		if (!list) return false;
		int i;
		ExprList::ExprListItem *item;
		for (i = 0, item = list->Ids; i < list->Exprs; i++, item++)
			if (FixExpr(item->Expr))
				return true;
		return false;
	}
#endif

#ifndef OMIT_TRIGGER
	__device__ bool DbFixer::FixTriggerStep(TriggerStep *step)
	{
		while (step)
		{
			if (FixSelect(step->Select) || FixExpr(step->Where) || FixExprList(step->ExprList))
				return true;
			step = step->Next;
		}
		return false;
	}
#endif

}