#ifndef CORE_EXT_H_
#define CORE_EXT_H_
#include "Core+Vdbe.cu.h"

typedef struct core_api_routines core_api_routines;

struct core_api_routines
{
	//void * (*aggregate_context)(FuncContext*,int nBytes);
	//int (*aggregate_count)(FuncContext*);
	//int (*bind_blob)(Vdbe*,int,const void*,int n,void(*)(void*));
	//int (*bind_double)(Vdbe*,int,double);
	//int (*bind_int)(Vdbe*,int,int);
	//int (*bind_int64)(Vdbe*,int,int64);
	//int (*bind_null)(Vdbe*,int);
	//int (*bind_parameter_count)(Vdbe*);
	//int (*bind_parameter_index)(Vdbe*,const char*zName);
	//const char * (*bind_parameter_name)(Vdbe*,int);
	//int (*bind_text)(Vdbe*,int,const char*,int n,void(*)(void*));
	//int (*bind_text16)(Vdbe*,int,const void*,int,void(*)(void*));
	//int (*bind_value)(Vdbe*,int,const Mem*);
	//int (*busy_handler)(Context*,int(*)(void*,int),void*);
	//int (*busy_timeout)(Context*,int ms);
	//int (*changes)(Context*);
	//int (*close)(Context*);
	//int (*collation_needed)(Context*,void*,void(*)(void*,Context*,int eTextRep,const char*));
	//int (*collation_needed16)(Context*,void*,void(*)(void*,Context*,int eTextRep,const void*));
	//const void * (*column_blob)(Vdbe*,int iCol);
	//int (*column_bytes)(Vdbe*,int iCol);
	//int (*column_bytes16)(Vdbe*,int iCol);
	//int (*column_count)(Vdbe*pStmt);
	//const char * (*column_database_name)(Vdbe*,int);
	//const void * (*column_database_name16)(Vdbe*,int);
	//const char * (*column_decltype)(Vdbe*,int i);
	//const void * (*column_decltype16)(Vdbe*,int);
	//double (*column_double)(Vdbe*,int iCol);
	//int (*column_int)(Vdbe*,int iCol);
	//int64 (*column_int64)(Vdbe*,int iCol);
	//const char * (*column_name)(Vdbe*,int);
	//const void * (*column_name16)(Vdbe*,int);
	//const char * (*column_origin_name)(Vdbe*,int);
	//const void * (*column_origin_name16)(Vdbe*,int);
	//const char * (*column_table_name)(Vdbe*,int);
	//const void * (*column_table_name16)(Vdbe*,int);
	//const unsigned char * (*column_text)(Vdbe*,int iCol);
	//const void * (*column_text16)(Vdbe*,int iCol);
	//int (*column_type)(Vdbe*,int iCol);
	//Mem* (*column_value)(Vdbe*,int iCol);
	//void * (*commit_hook)(Context*,int(*)(void*),void*);
	//int (*complete)(const char*sql);
	//int (*complete16)(const void*sql);
	//int (*create_collation)(Context*,const char*,int,void*, int(*)(void*,int,const void*,int,const void*));
	//int (*create_collation16)(Context*,const void*,int,void*, int(*)(void*,int,const void*,int,const void*));
	//int (*create_function)(Context*,const char*,int,int,void*, void (*xFunc)(FuncContext*,int,Mem**), void (*xStep)(FuncContext*,int,Mem**), void (*xFinal)(FuncContext*));
	//int (*create_function16)(Context*,const void*,int,int,void*, void (*xFunc)(FuncContext*,int,Mem**), void (*xStep)(FuncContext*,int,Mem**), void (*xFinal)(FuncContext*));
	//int (*create_module)(Context*,const char*,const ITableModule*,void*);
	//int (*data_count)(Vdbe*pStmt);
	//Context * (*db_handle)(Vdbe*);
	//int (*declare_vtab)(Context*,const char*);
	//int (*enable_shared_cache)(int);
	//int (*errcode)(Context*db);
	//const char * (*errmsg)(Context*);
	//const void * (*errmsg16)(Context*);
	//int (*exec)(Context*,const char*,callback,void*,char**);
	//int (*expired)(Vdbe*);
	//int (*finalize)(Vdbe*pStmt);
	//void (*free)(void*);
	//void (*free_table)(char**result);
	//int (*get_autocommit)(Context*);
	//void * (*get_auxdata)(FuncContext*,int);
	//int (*get_table)(Context*,const char*,char***,int*,int*,char**);
	//int (*global_recover)(void);
	//void (*interruptx)(Context*);
	//int64 (*last_insert_rowid)(Context*);
	//const char * (*libversion)(void);
	//int (*libversion_number)(void);
	//void *(*malloc)(int);
	//char * (*mprintf)(const char*,...);
	//int (*open)(const char*,Context**);
	//int (*open16)(const void*,Context**);
	//int (*prepare)(Context*,const char*,int,Vdbe**,const char**);
	//int (*prepare16)(Context*,const void*,int,Vdbe**,const void**);
	//void * (*profile)(Context*,void(*)(void*,const char*,uint64),void*);
	//void (*progress_handler)(Context*,int,int(*)(void*),void*);
	//void *(*realloc)(void*,int);
	//int (*reset)(Vdbe*pStmt);
	//void (*result_blob)(FuncContext*,const void*,int,void(*)(void*));
	//void (*result_double)(FuncContext*,double);
	//void (*result_error)(FuncContext*,const char*,int);
	//void (*result_error16)(FuncContext*,const void*,int);
	//void (*result_int)(FuncContext*,int);
	//void (*result_int64)(FuncContext*,int64);
	//void (*result_null)(FuncContext*);
	//void (*result_text)(FuncContext*,const char*,int,void(*)(void*));
	//void (*result_text16)(FuncContext*,const void*,int,void(*)(void*));
	//void (*result_text16be)(FuncContext*,const void*,int,void(*)(void*));
	//void (*result_text16le)(FuncContext*,const void*,int,void(*)(void*));
	//void (*result_value)(FuncContext*,Mem*);
	//void * (*rollback_hook)(Context*,void(*)(void*),void*);
	//int (*set_authorizer)(Context*,int(*)(void*,int,const char*,const char*,const char*,const char*),void*);
	//void (*set_auxdata)(FuncContext*,int,void*,void (*)(void*));
	//char * (*snprintf)(int,char*,const char*,...);
	//int (*step)(Vdbe*);
	//int (*table_column_metadata)(Context*,const char*,const char*,const char*,char const**,char const**,int*,int*,int*);
	//void (*thread_cleanup)(void);
	//int (*total_changes)(Context*);
	//void * (*trace)(Context*,void(*xTrace)(void*,const char*),void*);
	//int (*transfer_bindings)(Vdbe*,Vdbe*);
	//void * (*update_hook)(Context*,void(*)(void*,int ,char const*,char const*,int64),void*);
	//void * (*user_data)(FuncContext*);
	//const void * (*value_blob)(Mem*);
	//int (*value_bytes)(Mem*);
	//int (*value_bytes16)(Mem*);
	//double (*value_double)(Mem*);
	//int (*value_int)(Mem*);
	//int64 (*value_int64)(Mem*);
	//int (*value_numeric_type)(Mem*);
	//const unsigned char * (*value_text)(Mem*);
	//const void * (*value_text16)(Mem*);
	//const void * (*value_text16be)(Mem*);
	//const void * (*value_text16le)(Mem*);
	//int (*value_type)(Mem*);
	//char *(*vmprintf)(const char*,va_list);
	//// Added ???
	//int (*overload_function)(Context*, const char *zFuncName, int nArg);
	//// Added by 3.3.13
	//int (*prepare_v2)(Context*,const char*,int,Vdbe**,const char**);
	//int (*prepare16_v2)(Context*,const void*,int,Vdbe**,const void**);
	//int (*clear_bindings)(Vdbe*);
	//// Added by 3.4.1
	//int (*create_module_v2)(Context*,const char*,const ITableModule*,void*, void (*xDestroy)(void *));
	//// Added by 3.5.0
	//int (*bind_zeroblob)(Vdbe*,int,int);
	//int (*blob_bytes)(core_blob*);
	//int (*blob_close)(core_blob*);
	//int (*blob_open)(Context*,const char*,const char*,const char*,int64, int,core_blob**);
	//int (*blob_read)(core_blob*,void*,int,int);
	//int (*blob_write)(core_blob*,const void*,int,int);
	//int (*create_collation_v2)(Context*,const char*,int,void*, int(*)(void*,int,const void*,int,const void*), void(*)(void*));
	//int (*file_control)(Context*,const char*,int,void*);
	//int64 (*memory_highwater)(int);
	//int64 (*memory_used)(void);
	//MutexEx *(*mutex_alloc)(int);
	//void (*mutex_enter)(MutexEx*);
	//void (*mutex_free)(MutexEx*);
	//void (*mutex_leave)(MutexEx*);
	//int (*mutex_try)(MutexEx*);
	//int (*open_v2)(const char*,Context**,int,const char*);
	//int (*release_memory)(int);
	//void (*result_error_nomem)(FuncContext*);
	//void (*result_error_toobig)(FuncContext*);
	//int (*sleep)(int);
	//void (*soft_heap_limit)(int);
	//VSystem *(*vfs_find)(const char*);
	//int (*vfs_register)(VSystem*,int);
	//int (*vfs_unregister)(VSystem*);
	//int (*xthreadsafe)(void);
	//void (*result_zeroblob)(FuncContext*,int);
	//void (*result_error_code)(FuncContext*,int);
	//int (*test_control)(int, ...);
	//void (*randomness)(int,void*);
	//Context *(*context_db_handle)(FuncContext*);
	//int (*extended_result_codes)(Context*,int);
	//int (*limit)(Context*,int,int);
	//Vdbe *(*next_stmt)(Context*,Vdbe*);
	//const char *(*sql)(Vdbe*);
	//int (*status)(int,int*,int*,int);
	//int (*backup_finish)(core_backup*);
	//core_backup *(*backup_init)(Context*,const char*,Context*,const char*);
	//int (*backup_pagecount)(core_backup*);
	//int (*backup_remaining)(core_backup*);
	//int (*backup_step)(core_backup*,int);
	//const char *(*compileoption_get)(int);
	//int (*compileoption_used)(const char*);
	//int (*create_function_v2)(Context*,const char*,int,int,void*, void (*xFunc)(FuncContext*,int,Mem**), void (*xStep)(FuncContext*,int,Mem**), void (*xFinal)(FuncContext*), void(*xDestroy)(void*));
	//int (*db_config)(Context*,int,...);
	//MutexEx *(*db_mutex)(Context*);
	//int (*db_status)(Context*,int,int*,int*,int);
	//int (*extended_errcode)(Context*);
	//void (*log)(int,const char*,...);
	//int64 (*soft_heap_limit64)(int64);
	//const char *(*sourceid)(void);
	//int (*stmt_status)(Vdbe*,int,int);
	//int (*strnicmp)(const char*,const char*,int);
	//int (*unlock_notify)(Context*,void(*)(void**,int),void*);
	//int (*wal_autocheckpoint)(Context*,int);
	//int (*wal_checkpoint)(Context*,const char*);
	//void *(*wal_hook)(Context*,int(*)(void*,Context*,const char*,int),void*);
	//int (*blob_reopen)(core_blob*,int64);
	//int (*vtab_config)(Context*,int op,...);
	//int (*vtab_on_conflict)(Context*);
	///* Version 3.7.16 and later */
	//int (*close_v2)(Context*);
	//const char *(*db_filename)(Context*,const char*);
	//int (*db_readonly)(Context*,const char*);
	//int (*db_release_memory)(Context*);
	//const char *(*errstr)(int);
	//int (*stmt_busy)(Vdbe*);
	//int (*stmt_readonly)(Vdbe*);
	//int (*stricmp)(const char*,const char*);
	//int (*uri_boolean)(const char*,const char*,int);
	//int64 (*uri_int64)(const char*,const char*,int64);
	//const char *(*uri_parameter)(const char*,const char*);
	//char *(*vsnprintf)(int,char*,const char*,va_list);
	//int (*wal_checkpoint_v2)(Context*,const char*,int,int*,int*);
};

#ifndef APICORE
//#define core_aggregate_context      g_api->aggregate_context
//#ifndef OMIT_DEPRECATED
//#define core_aggregate_count        g_api->aggregate_count
//#endif
//#define core_bind_blob              g_api->bind_blob
//#define core_bind_double            g_api->bind_double
//#define core_bind_int               g_api->bind_int
//#define core_bind_int64             g_api->bind_int64
//#define core_bind_null              g_api->bind_null
//#define core_bind_parameter_count   g_api->bind_parameter_count
//#define core_bind_parameter_index   g_api->bind_parameter_index
//#define core_bind_parameter_name    g_api->bind_parameter_name
//#define core_bind_text              g_api->bind_text
//#define core_bind_text16            g_api->bind_text16
//#define core_bind_value             g_api->bind_value
//#define core_busy_handler           g_api->busy_handler
//#define core_busy_timeout           g_api->busy_timeout
//#define core_changes                g_api->changes
//#define core_close                  g_api->close
//#define core_collation_needed       g_api->collation_needed
//#define core_collation_needed16     g_api->collation_needed16
//#define core_column_blob            g_api->column_blob
//#define core_column_bytes           g_api->column_bytes
//#define core_column_bytes16         g_api->column_bytes16
//#define core_column_count           g_api->column_count
//#define core_column_database_name   g_api->column_database_name
//#define core_column_database_name16 g_api->column_database_name16
//#define core_column_decltype        g_api->column_decltype
//#define core_column_decltype16      g_api->column_decltype16
//#define core_column_double          g_api->column_double
//#define core_column_int             g_api->column_int
//#define core_column_int64           g_api->column_int64
//#define core_column_name            g_api->column_name
//#define core_column_name16          g_api->column_name16
//#define core_column_origin_name     g_api->column_origin_name
//#define core_column_origin_name16   g_api->column_origin_name16
//#define core_column_table_name      g_api->column_table_name
//#define core_column_table_name16    g_api->column_table_name16
//#define core_column_text            g_api->column_text
//#define core_column_text16          g_api->column_text16
//#define core_column_type            g_api->column_type
//#define core_column_value           g_api->column_value
//#define core_commit_hook            g_api->commit_hook
//#define core_complete               g_api->complete
//#define core_complete16             g_api->complete16
//#define core_create_collation       g_api->create_collation
//#define core_create_collation16     g_api->create_collation16
//#define core_create_function        g_api->create_function
//#define core_create_function16      g_api->create_function16
//#define core_create_module          g_api->create_module
//#define core_create_module_v2       g_api->create_module_v2
//#define core_data_count             g_api->data_count
//#define core_db_handle              g_api->db_handle
//#define core_declare_vtab           g_api->declare_vtab
//#define core_enable_shared_cache    g_api->enable_shared_cache
//#define core_errcode                g_api->errcode
//#define core_errmsg                 g_api->errmsg
//#define core_errmsg16               g_api->errmsg16
//#define core_exec                   g_api->exec
//#ifndef OMIT_DEPRECATED
//#define core_expired                g_api->expired
//#endif
//#define core_finalize               g_api->finalize
//#define core_free                   g_api->free
//#define core_free_table             g_api->free_table
//#define core_get_autocommit         g_api->get_autocommit
//#define core_get_auxdata            g_api->get_auxdata
//#define core_get_table              g_api->get_table
//#ifndef OMIT_DEPRECATED
//#define core_global_recover         g_api->global_recover
//#endif
//#define core_interrupt              g_api->interruptx
//#define core_last_insert_rowid      g_api->last_insert_rowid
//#define core_libversion             g_api->libversion
//#define core_libversion_number      g_api->libversion_number
//#define core_malloc                 g_api->malloc
//#define core_mprintf                g_api->mprintf
//#define core_open                   g_api->open
//#define core_open16                 g_api->open16
//#define core_prepare                g_api->prepare
//#define core_prepare16              g_api->prepare16
//#define core_prepare_v2             g_api->prepare_v2
//#define core_prepare16_v2           g_api->prepare16_v2
//#define core_profile                g_api->profile
//#define core_progress_handler       g_api->progress_handler
//#define core_realloc                g_api->realloc
//#define core_reset                  g_api->reset
//#define core_result_blob            g_api->result_blob
//#define core_result_double          g_api->result_double
//#define core_result_error           g_api->result_error
//#define core_result_error16         g_api->result_error16
//#define core_result_int             g_api->result_int
//#define core_result_int64           g_api->result_int64
//#define core_result_null            g_api->result_null
//#define core_result_text            g_api->result_text
//#define core_result_text16          g_api->result_text16
//#define core_result_text16be        g_api->result_text16be
//#define core_result_text16le        g_api->result_text16le
//#define core_result_value           g_api->result_value
//#define core_rollback_hook          g_api->rollback_hook
//#define core_set_authorizer         g_api->set_authorizer
//#define core_set_auxdata            g_api->set_auxdata
//#define core_snprintf               g_api->snprintf
//#define core_step                   g_api->step
//#define core_table_column_metadata  g_api->table_column_metadata
//#define core_thread_cleanup         g_api->thread_cleanup
//#define core_total_changes          g_api->total_changes
//#define core_trace                  g_api->trace
//#ifndef OMIT_DEPRECATED
//#define core_transfer_bindings      g_api->transfer_bindings
//#endif
//#define core_update_hook            g_api->update_hook
//#define core_user_data              g_api->user_data
//#define core_value_blob             g_api->value_blob
//#define core_value_bytes            g_api->value_bytes
//#define core_value_bytes16          g_api->value_bytes16
//#define core_value_double           g_api->value_double
//#define core_value_int              g_api->value_int
//#define core_value_int64            g_api->value_int64
//#define core_value_numeric_type     g_api->value_numeric_type
//#define core_value_text             g_api->value_text
//#define core_value_text16           g_api->value_text16
//#define core_value_text16be         g_api->value_text16be
//#define core_value_text16le         g_api->value_text16le
//#define core_value_type             g_api->value_type
//#define core_vmprintf               g_api->vmprintf
//#define core_overload_function      g_api->overload_function
//#define core_prepare_v2             g_api->prepare_v2
//#define core_prepare16_v2           g_api->prepare16_v2
//#define core_clear_bindings         g_api->clear_bindings
//#define core_bind_zeroblob          g_api->bind_zeroblob
//#define core_blob_bytes             g_api->blob_bytes
//#define core_blob_close             g_api->blob_close
//#define core_blob_open              g_api->blob_open
//#define core_blob_read              g_api->blob_read
//#define core_blob_write             g_api->blob_write
//#define core_create_collation_v2    g_api->create_collation_v2
//#define core_file_control           g_api->file_control
//#define core_memory_highwater       g_api->memory_highwater
//#define core_memory_used            g_api->memory_used
//#define core_mutex_alloc            g_api->mutex_alloc
//#define core_mutex_enter            g_api->mutex_enter
//#define core_mutex_free             g_api->mutex_free
//#define core_mutex_leave            g_api->mutex_leave
//#define core_mutex_try              g_api->mutex_try
//#define core_open_v2                g_api->open_v2
//#define core_release_memory         g_api->release_memory
//#define core_result_error_nomem     g_api->result_error_nomem
//#define core_result_error_toobig    g_api->result_error_toobig
//#define core_sleep                  g_api->sleep
//#define core_soft_heap_limit        g_api->soft_heap_limit
//#define core_vfs_find               g_api->vfs_find
//#define core_vfs_register           g_api->vfs_register
//#define core_vfs_unregister         g_api->vfs_unregister
//#define core_threadsafe             g_api->xthreadsafe
//#define core_result_zeroblob        g_api->result_zeroblob
//#define core_result_error_code      g_api->result_error_code
//#define core_test_control           g_api->test_control
//#define core_randomness             g_api->randomness
//#define core_context_db_handle      g_api->context_db_handle
//#define core_extended_result_codes  g_api->extended_result_codes
//#define core_limit                  g_api->limit
//#define core_next_stmt              g_api->next_stmt
//#define core_sql                    g_api->sql
//#define core_status                 g_api->status
//#define core_backup_finish          g_api->backup_finish
//#define core_backup_init            g_api->backup_init
//#define core_backup_pagecount       g_api->backup_pagecount
//#define core_backup_remaining       g_api->backup_remaining
//#define core_backup_step            g_api->backup_step
//#define core_compileoption_get      g_api->compileoption_get
//#define core_compileoption_used     g_api->compileoption_used
//#define core_create_function_v2     g_api->create_function_v2
//#define core_db_config              g_api->db_config
//#define core_db_mutex               g_api->db_mutex
//#define core_db_status              g_api->db_status
//#define core_extended_errcode       g_api->extended_errcode
//#define core_log                    g_api->log
//#define core_soft_heap_limit64      g_api->soft_heap_limit64
//#define core_sourceid               g_api->sourceid
//#define core_stmt_status            g_api->stmt_status
//#define core_strnicmp               g_api->strnicmp
//#define core_unlock_notify          g_api->unlock_notify
//#define core_wal_autocheckpoint     g_api->wal_autocheckpoint
//#define core_wal_checkpoint         g_api->wal_checkpoint
//#define core_wal_hook               g_api->wal_hook
//#define core_blob_reopen            g_api->blob_reopen
//#define core_vtab_config            g_api->vtab_config
//#define core_vtab_on_conflict       g_api->vtab_on_conflict
//// Version 3.7.16 and later
//#define core_close_v2               g_api->close_v2
//#define core_db_filename            g_api->db_filename
//#define core_db_readonly            g_api->db_readonly
//#define core_db_release_memory      g_api->db_release_memory
//#define core_errstr                 g_api->errstr
//#define core_stmt_busy              g_api->stmt_busy
//#define core_stmt_readonly          g_api->stmt_readonly
//#define core_stricmp                g_api->stricmp
//#define core_uri_boolean            g_api->uri_boolean
//#define core_uri_int64              g_api->uri_int64
//#define core_uri_parameter          g_api->uri_parameter
//#define core_uri_vsnprintf          g_api->vsnprintf
//#define core_wal_checkpoint_v2      g_api->wal_checkpoint_v2
#endif

#define EXTENSION_INIT1     const core_api_routines *g_api = nullptr;
#define EXTENSION_INIT2(v)  g_api = v;

#endif
