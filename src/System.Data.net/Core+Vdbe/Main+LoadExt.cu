#ifndef APICORE
#define APICORE 1 // Disable the API redefinition in sqlite3ext.h
#endif
#include "Core+Ext.cu.h"
#include <string.h>

namespace Core
{

#pragma region OMIT_LOAD_EXTENSION
#ifndef OMIT_LOAD_EXTENSION

	//#ifndef ENABLE_COLUMN_METADATA
	//#define core_column_database_name   nullptr
	//#define core_column_database_name16 nullptr
	//#define core_column_table_name      nullptr
	//#define core_column_table_name16    nullptr
	//#define core_column_origin_name     nullptr
	//#define core_column_origin_name16   nullptr
	//#define core_table_column_metadata  nullptr
	//#endif
	//#ifdef OMIT_AUTHORIZATION
	//#define core_set_authorizer         nullptr
	//#endif
	//#ifdef OMIT_UTF16
	//#define core_bind_text16            nullptr
	//#define core_collation_needed16     nullptr
	//#define core_column_decltype16      nullptr
	//#define core_column_name16          nullptr
	//#define core_column_text16          nullptr
	//#define core_complete16             nullptr
	//#define core_create_collation16     nullptr
	//#define core_create_function16      nullptr
	//#define core_errmsg16               nullptr
	//#define core_open16                 nullptr
	//#define core_prepare16              nullptr
	//#define core_prepare16_v2           nullptr
	//#define core_result_error16         nullptr
	//#define core_result_text16          nullptr
	//#define core_result_text16be        nullptr
	//#define core_result_text16le        nullptr
	//#define core_value_text16           nullptr
	//#define core_value_text16be         nullptr
	//#define core_value_text16le         nullptr
	//#define core_column_database_name16 nullptr
	//#define core_column_table_name16    nullptr
	//#define core_column_origin_name16	nullptr
	//#endif
	//#ifdef OMIT_COMPLETE
	//#define core_complete				nullptr
	//#define core_complete16				nullptr
	//#endif
	//#ifdef OMIT_DECLTYPE
	//#define core_column_decltype16		nullptr
	//#define core_column_decltype		nullptr
	//#endif
	//#ifdef OMIT_PROGRESS_CALLBACK
	//#define core_progress_handler		nullptr
	//#endif
	//#ifdef OMIT_VIRTUALTABLE
	//#define core_create_module			nullptr
	//#define core_create_module_v2		nullptr
	//#define core_declare_vtab			nullptr
	//#define core_vtab_config			nullptr
	//#define core_vtab_on_conflict		nullptr
	//#endif
	//#ifdef OMIT_SHARED_CACHE
	//#define core_enable_shared_cache	nullptr
	//#endif
	//#ifdef OMIT_TRACE
	//#define core_profile				nullptr
	//#define core_trace					nullptr
	//#endif
	//#ifdef OMIT_GET_TABLE
	//#define core_free_table				nullptr
	//#define core_get_table				nullptr
	//#endif
	//#ifdef OMIT_INCRBLOB
	//#define core_bind_zeroblob			nullptr
	//#define core_blob_bytes				nullptr
	//#define core_blob_close				nullptr
	//#define core_blob_open				nullptr
	//#define core_blob_read				nullptr
	//#define core_blob_write				nullptr
	//#define core_blob_reopen			nullptr
	//#endif

	__constant__ static const core_api_routines g_apis =
	{
		//		core_aggregate_context,
		//#ifndef OMIT_DEPRECATED
		//		core_aggregate_count,
		//#else
		//		nullptr,
		//#endif
		//		core_bind_blob,
		//		core_bind_double,
		//		core_bind_int,
		//		core_bind_int64,
		//		core_bind_null,
		//		core_bind_parameter_count,
		//		core_bind_parameter_index,
		//		core_bind_parameter_name,
		//		core_bind_text,
		//		core_bind_text16,
		//		core_bind_value,
		//		core_busy_handler,
		//		core_busy_timeout,
		//		core_changes,
		//		core_close,
		//		core_collation_needed,
		//		core_collation_needed16,
		//		core_column_blob,
		//		core_column_bytes,
		//		core_column_bytes16,
		//		core_column_count,
		//		core_column_database_name,
		//		core_column_database_name16,
		//		core_column_decltype,
		//		core_column_decltype16,
		//		core_column_double,
		//		core_column_int,
		//		core_column_int64,
		//		core_column_name,
		//		core_column_name16,
		//		core_column_origin_name,
		//		core_column_origin_name16,
		//		core_column_table_name,
		//		core_column_table_name16,
		//		core_column_text,
		//		core_column_text16,
		//		core_column_type,
		//		core_column_value,
		//		core_commit_hook,
		//		core_complete,
		//		core_complete16,
		//		core_create_collation,
		//		core_create_collation16,
		//		core_create_function,
		//		core_create_function16,
		//		core_create_module,
		//		core_data_count,
		//		core_db_handle,
		//		core_declare_vtab,
		//		core_enable_shared_cache,
		//		core_errcode,
		//		core_errmsg,
		//		core_errmsg16,
		//		core_exec,
		//#ifndef OMIT_DEPRECATED
		//		core_expired,
		//#else
		//		nullptr,
		//#endif
		//		core_finalize,
		//		core_free,
		//		core_free_table,
		//		core_get_autocommit,
		//		core_get_auxdata,
		//		core_get_table,
		//		nullptr, // Was core_global_recover(), but that function is deprecated
		//		core_interrupt,
		//		core_last_insert_rowid,
		//		core_libversion,
		//		core_libversion_number,
		//		core_malloc,
		//		core_mprintf,
		//		core_open,
		//		core_open16,
		//		core_prepare,
		//		core_prepare16,
		//		core_profile,
		//		core_progress_handler,
		//		core_realloc,
		//		core_reset,
		//		core_result_blob,
		//		core_result_double,
		//		core_result_error,
		//		core_result_error16,
		//		core_result_int,
		//		core_result_int64,
		//		core_result_null,
		//		core_result_text,
		//		core_result_text16,
		//		core_result_text16be,
		//		core_result_text16le,
		//		core_result_value,
		//		core_rollback_hook,
		//		core_set_authorizer,
		//		core_set_auxdata,
		//		core_snprintf,
		//		core_step,
		//		core_table_column_metadata,
		//#ifndef OMIT_DEPRECATED
		//		core_thread_cleanup,
		//#else
		//		nullptr,
		//#endif
		//		core_total_changes,
		//		core_trace,
		//#ifndef OMIT_DEPRECATED
		//		core_transfer_bindings,
		//#else
		//		nullptr,
		//#endif
		//		core_update_hook,
		//		core_user_data,
		//		core_value_blob,
		//		core_value_bytes,
		//		core_value_bytes16,
		//		core_value_double,
		//		core_value_int,
		//		core_value_int64,
		//		core_value_numeric_type,
		//		core_value_text,
		//		core_value_text16,
		//		core_value_text16be,
		//		core_value_text16le,
		//		core_value_type,
		//		core_vmprintf,
		//		// *************************************************************************
		//		core_overload_function,
		//
		//		// Added after 3.3.13
		//		core_prepare_v2,
		//		core_prepare16_v2,
		//		core_clear_bindings,
		//
		//		// Added for 3.4.1
		//		core_create_module_v2,
		//
		//		// Added for 3.5.0
		//		core_bind_zeroblob,
		//		core_blob_bytes,
		//		core_blob_close,
		//		core_blob_open,
		//		core_blob_read,
		//		core_blob_write,
		//		core_create_collation_v2,
		//		core_file_control,
		//		core_memory_highwater,
		//		core_memory_used,
		//#ifdef MUTEX_OMIT
		//		nullptr, 
		//		nullptr, 
		//		nullptr,
		//		nullptr,
		//		nullptr,
		//#else
		//		core_mutex_alloc,
		//		core_mutex_enter,
		//		core_mutex_free,
		//		core_mutex_leave,
		//		core_mutex_try,
		//#endif
		//		core_open_v2,
		//		core_release_memory,
		//		core_result_error_nomem,
		//		core_result_error_toobig,
		//		core_sleep,
		//		core_soft_heap_limit,
		//		core_vfs_find,
		//		core_vfs_register,
		//		core_vfs_unregister,
		//
		//		// Added for 3.5.8
		//		core_threadsafe,
		//		core_result_zeroblob,
		//		core_result_error_code,
		//		core_test_control,
		//		core_randomness,
		//		core_context_db_handle,
		//
		//		// Added for 3.6.0
		//		core_extended_result_codes,
		//		core_limit,
		//		core_next_stmt,
		//		core_sql,
		//		core_status,
		//
		//		// Added for 3.7.4
		//		core_backup_finish,
		//		core_backup_init,
		//		core_backup_pagecount,
		//		core_backup_remaining,
		//		core_backup_step,
		//#ifndef OMIT_COMPILEOPTION_DIAGS
		//		core_compileoption_get,
		//		core_compileoption_used,
		//#else
		//		nullptr,
		//		nullptr,
		//#endif
		//		core_create_function_v2,
		//		core_db_config,
		//		core_db_mutex,
		//		core_db_status,
		//		core_extended_errcode,
		//		core_log,
		//		core_soft_heap_limit64,
		//		core_sourceid,
		//		core_stmt_status,
		//		core_strnicmp,
		//#ifdef ENABLE_UNLOCK_NOTIFY
		//		core_unlock_notify,
		//#else
		//		nullptr,
		//#endif
		//#ifndef OMIT_WAL
		//		core_wal_autocheckpoint,
		//		core_wal_checkpoint,
		//		core_wal_hook,
		//#else
		//		nullptr,
		//		nullptr,
		//		nullptr,
		//#endif
		//		core_blob_reopen,
		//		core_vtab_config,
		//		core_vtab_on_conflict,
		//		core_close_v2,
		//		core_db_filename,
		//		core_db_readonly,
		//		core_db_release_memory,
		//		core_errstr,
		//		core_stmt_busy,
		//		core_stmt_readonly,
		//		core_stricmp,
		//		core_uri_boolean,
		//		core_uri_int64,
		//		core_uri_parameter,
		//		core_vsnprintf,
		//		core_wal_checkpoint_v2
	};

	__device__ RC Main::LoadExtension_(Context *ctx, const char *fileName, const char *procName, char **errMsgOut)
	{
		if (errMsgOut) *errMsgOut = nullptr;

		// Ticket #1863.  To avoid a creating security problems for older applications that relink against newer versions of SQLite, the
		// ability to run load_extension is turned off by default.  One must call core_enable_load_extension() to turn on extension
		// loading.  Otherwise you get the following error.
		if ((ctx->Flags & Context::FLAG_LoadExtension) == 0)
		{
			if (errMsgOut)
				*errMsgOut = _mprintf("not authorized");
			return RC_ERROR;
		}

		if (!procName)
			procName = "core_extension_init";

		VSystem *vfs = ctx->Vfs;
		void *handle = vfs->DlOpen(fileName);
		char *errmsg = nullptr;
		int msgLength = 300 + _strlen30(fileName);
		if (!handle)
		{
			if (errMsgOut)
			{
				*errMsgOut = errmsg = (char *)_alloc(msgLength);
				if (errmsg)
				{
					__snprintf(errmsg, msgLength, "unable to open shared library [%s]", fileName);
					vfs->DlError(msgLength-1, errmsg);
				}
			}
			return RC_ERROR;
		}

		RC (*init)(Context*,char**,const core_api_routines*) = (RC(*)(Context*,char**,const core_api_routines*))vfs->DlSym(handle, procName);
		if (!init)
		{
			if (errMsgOut)
			{
				msgLength += _strlen30(procName);
				*errMsgOut = errmsg = (char *)_alloc(msgLength);
				if (errmsg)
				{
					__snprintf(errmsg, msgLength, "no entry point [%s] in shared library [%s]", procName, fileName);
					vfs->DlError(msgLength-1, errmsg);
				}
				vfs->DlClose(handle);
			}
			return RC_ERROR;
		}
		else if (init(ctx, &errmsg, &g_apis))
		{
			if (errMsgOut)
				*errMsgOut = _mprintf("error during initialization: %s", errmsg);
			_free(errmsg);
			vfs->DlClose(handle);
			return RC_ERROR;
		}

		// Append the new shared library handle to the ctx->aExtension array.
		void **handles = (void **)_tagalloc2(ctx, sizeof(handle)*(ctx->Extensions.length+1), true);
		if (!handles)
			return RC_NOMEM;
		if (ctx->Extensions.length > 0)
			_memcpy(handles, ctx->Extensions.data, sizeof(handle)*ctx->Extensions.length);
		_tagfree(ctx, ctx->Extensions.data);
		ctx->Extensions.data = handles;

		ctx->Extensions[ctx->Extensions.length++] = handle;
		return RC_OK;
	}

	__device__ RC Main::LoadExtension(Context *ctx, const char *fileName, const char *procName, char **errMsgOut)
	{
		MutexEx::Enter(ctx->Mutex);
		RC rc = LoadExtension_(ctx, fileName, procName, errMsgOut);
		rc = ApiExit(ctx, rc);
		MutexEx::Leave(ctx->Mutex);
		return rc;
	}

	__device__ void Main::CloseExtensions(Context *ctx)
	{
		_assert(MutexEx::Held(ctx->Mutex));
		for (int i = 0; i < ctx->Extensions.length; i++)
			ctx->Vfs->DlClose(ctx->Extensions[i]);
		_tagfree(ctx, ctx->Extensions.data);
	}

	__device__ RC Main::EnableLoadExtension(Context *ctx, bool onoff)
	{
		MutexEx::Enter(ctx->Mutex);
		if (onoff)
			ctx->Flags |= Context::FLAG_LoadExtension;
		else
			ctx->Flags &= ~Context::FLAG_LoadExtension;
		MutexEx::Leave(ctx->Mutex);
		return RC_OK;
	}

#else
	static const core_api_routines g_apis = { 0 };
#endif
#pragma endregion

	typedef struct AutoExtList_t AutoExtList_t;
	__device__ static _WSD struct AutoExtList_t
	{
		int ExtsLength;     // Number of entries in aExt[]
		void (**Exts)();	// Pointers to the extension init functions
	} g_autoext = { 0, nullptr };

#ifdef OMIT_WSD
#define WsdAutoextInit AutoExtList_t *x = &_GLOBAL(AutoExtList_t, g_autoext)
#define WsdAutoext x[0]
#else
#define WsdAutoextInit
#define WsdAutoext g_autoext
#endif

	__device__ RC Main::AutoExtension(void (*init)())
	{
		RC rc = RC_OK;
#ifndef OMIT_AUTOINIT
		rc = Initialize();
		if (rc)
			return rc;
		else
#endif
		{
#if THREADSAFE
			MutexEx mutex = MutexEx::Alloc(MutexEx::MUTEX_STATIC_MASTER);
#endif
			WsdAutoextInit;
			MutexEx::Enter(mutex);
			int i;
			for (i = 0; i < WsdAutoext.ExtsLength; i++)
				if (WsdAutoext.Exts[i] == init) break;
			if (i == WsdAutoext.ExtsLength)
			{
				int bytes = (WsdAutoext.ExtsLength + 1)*sizeof(WsdAutoext.Exts[0]);
				void (**newExts)() = (void (**)())_realloc(WsdAutoext.Exts, bytes);
				if (!newExts)
					rc = RC_NOMEM;
				else
				{
					WsdAutoext.Exts = newExts;
					WsdAutoext.Exts[WsdAutoext.ExtsLength] = init;
					WsdAutoext.ExtsLength++;
				}
			}
			MutexEx::Leave(mutex);
			_assert((rc & 0xff) == rc);
			return rc;
		}
	}

	__device__ void Main::ResetAutoExtension()
	{
#ifndef OMIT_AUTOINIT
		if (Initialize() == RC_OK)
#endif
		{
#if THREADSAFE
			MutexEx mutex = MutexEx::Alloc(MutexEx::MUTEX_STATIC_MASTER);
#endif
			WsdAutoextInit;
			MutexEx::Enter(mutex);
			_free(WsdAutoext.Exts);
			WsdAutoext.Exts = nullptr;
			WsdAutoext.ExtsLength = 0;
			MutexEx::Leave(mutex);
		}
	}

	__device__ void Main::AutoLoadExtensions(Context *ctx)
	{
		WsdAutoextInit;
		if (WsdAutoext.ExtsLength == 0)
			return; // Common case: early out without every having to acquire a mutex
		bool go = true;
		for (int i = 0; go; i++)
		{
			char *errmsg;
#if THREADSAFE
			MutexEx mutex = MutexEx::Alloc(MutexEx::MUTEX_STATIC_MASTER);
#endif
			MutexEx::Enter(mutex);
			RC (*init)(Context*,char**,const core_api_routines*);
			if (i >= WsdAutoext.ExtsLength)
			{
				init = nullptr;
				go = false;
			}
			else
				init = (RC(*)(Context*,char**,const core_api_routines*))WsdAutoext.Exts[i];
			MutexEx::Leave(mutex);
			errmsg = nullptr;
			RC rc;
			if (init && (rc = init(ctx, &errmsg, &g_apis)) != 0)
			{
				Error(ctx, rc, "automatic extension loading failed: %s", errmsg);
				go = false;
			}
			_free(errmsg);
		}
	}
}