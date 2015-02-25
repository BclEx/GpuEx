using System;
using System.Diagnostics;
using HANDLE = System.IntPtr;
using System.Text;

namespace Core
{
    public partial class Main
    {
#if !APICORE
        const int APICORE = 1; // Disable the API redefinition in sqlite3ext.h
#endif

        #region OMIT_LOAD_EXTENSION
#if !OMIT_LOAD_EXTENSION

#if !ENABLE_COLUMN_METADATA
#endif
#if OMIT_AUTHORIZATION
#endif
#if OMIT_UTF16
    static string sqlite3_errmsg16( sqlite3 db ) { return ""; }
    static void sqlite3_result_text16( sqlite3_context pCtx, string z, int n, dxDel xDel ) { }
#endif
#if OMIT_COMPLETE
#endif
#if OMIT_DECLTYPE
#endif
#if OMIT_PROGRESS_CALLBACK
static void sqlite3_progress_handler(sqlite3 db,       int nOps, dxProgress xProgress, object pArg){}
#endif
#if OMIT_VIRTUALTABLE
#endif
#if OMIT_SHARED_CACHE
#endif
#if OMIT_TRACE
#endif
#if OMIT_GET_TABLE
    public static int sqlite3_get_table(sqlite3 db, string zSql, ref string[] pazResult, ref int pnRow, ref int pnColumn, ref string pzErrmsg) { return 0; }
#endif
#if OMIT_INCRBLOB
#endif

        public class core_api_routines
        {
            public Context context_db_handle;
        }

        static core_api_routines g_apis = new core_api_routines();

        public static RC LoadExtension_(Context ctx, string fileName, string procName, ref string errMsgOut)
        {
            if (errMsgOut != null) errMsgOut = null;

            // Ticket #1863.  To avoid a creating security problems for older applications that relink against newer versions of SQLite, the
            // ability to run load_extension is turned off by default.  One must call core_enable_load_extension() to turn on extension
            // loading.  Otherwise you get the following error.
            if ((ctx.Flags & Context.FLAG.LoadExtension) == 0)
            {
                errMsgOut = C._mprintf("not authorized");
                return RC.ERROR;
            }

            if (procName == null)
                procName = "sqlite3_extension_init";

            VSystem vfs = ctx.Vfs;
            HANDLE handle = (HANDLE)vfs.DlOpen(fileName);
            StringBuilder errmsg = new StringBuilder(100);
            int msgLength = 300;

            if (handle == IntPtr.Zero)
            {
                errMsgOut = string.Empty;
                C.__snprintf(errmsg, msgLength, "unable to open shared library [%s]", fileName);
                vfs.DlError(msgLength - 1, errmsg.ToString());
                return RC.ERROR;
            }
            Func<Context, StringBuilder, core_api_routines, RC> init = (Func<Context, StringBuilder, core_api_routines, RC>)vfs.DlSym(handle, procName);
            Debugger.Break();
            if (init == null)
            {
                msgLength += procName.Length;
                C.__snprintf(errmsg, msgLength, "no entry point [%s] in shared library [%s]", procName, fileName);
                vfs.DlError(msgLength - 1, errMsgOut = errmsg.ToString());
                vfs.DlClose(handle);
                return RC.ERROR;
            }
            else if (init(ctx, errmsg, g_apis) != 0)
            {
                errMsgOut = C._mprintf("error during initialization: %s", errmsg.ToString());
                C._tagfree(ctx, ref errmsg);
                vfs.DlClose(handle);
                return RC.ERROR;
            }

            // Append the new shared library handle to the db.aExtension array.
            object[] handles = new object[ctx.Extensions.length + 1];
            if (handles == null)
                return RC.NOMEM;
            if (ctx.Extensions.length > 0)
                Array.Copy(ctx.Extensions.data, handles, ctx.Extensions.length);
            C._tagfree(ctx, ref ctx.Extensions.data);
            ctx.Extensions.data = handles;

            ctx.Extensions[ctx.Extensions.length++] = handle;
            return RC.OK;
        }

        public static RC LoadExtension(Context ctx, string fileName, string procName, ref string errMsg)
        {
            MutexEx.Enter(ctx.Mutex);
            RC rc = LoadExtension_(ctx, fileName, procName, ref errMsg);
            rc = ApiExit(ctx, rc);
            MutexEx.Leave(ctx.Mutex);
            return rc;
        }

        public static void CloseExtensions(Context ctx)
        {
            Debug.Assert(MutexEx.Held(ctx.Mutex));
            for (int i = 0; i < ctx.Extensions.length; i++)
                ctx.Vfs.DlClose((HANDLE)ctx.Extensions[i]);
            C._tagfree(ctx, ref ctx.Extensions.data);
        }

        public static RC EnableLoadExtension(Context ctx, bool onoff)
        {
            MutexEx.Enter(ctx.Mutex);
            if (onoff)
                ctx.Flags |= Context.FLAG.LoadExtension;
            else
                ctx.Flags &= ~Context.FLAG.LoadExtension;
            MutexEx.Leave(ctx.Mutex);
            return RC.OK;
        }
#else
        const core_api_routines g_apis = null;
#endif
        #endregion

        public class AutoExtList_t
        {
            public int ExtsLength = 0;      // Number of entries in aExt[]
            public Func<Context, string, core_api_routines, RC>[] Exts = null;    // Pointers to the extension init functions
            public AutoExtList_t(int extsLength, Func<Context, string, core_api_routines, RC>[] exts)
            {
                ExtsLength = extsLength;
                Exts = exts;
            }
        }
        static AutoExtList_t g_autoext = new AutoExtList_t(0, null);

        static RC AutoExtension(Func<Context, string, core_api_routines, RC> init)
        {
            RC rc = RC.OK;
#if !OMIT_AUTOINIT
            rc = Initialize();
            if (rc != 0)
                return rc;
            else
#endif
            {
#if THREADSAFE
                MutexEx mutex = MutexEx.Alloc(MutexEx.MUTEX.STATIC_MASTER);
#endif
                MutexEx.Enter(mutex);
                int i;
                for (i = 0; i < g_autoext.ExtsLength; i++)
                    if (g_autoext.Exts[i] == init) break;
                if (i == g_autoext.ExtsLength)
                {
                    Array.Resize(ref g_autoext.Exts, g_autoext.ExtsLength + 1);
                    g_autoext.Exts[g_autoext.ExtsLength] = init;
                    g_autoext.ExtsLength++;
                }
                MutexEx.Leave(mutex);
                Debug.Assert((rc & (RC)0xff) == rc);
                return rc;
            }
        }

        public static void ResetAutoExtension()
        {
#if !OMIT_AUTOINIT
            if (Initialize() == RC.OK)
#endif
            {
#if THREADSAFE
                MutexEx mutex = MutexEx.Alloc(MutexEx.MUTEX.STATIC_MASTER);
#endif
                MutexEx.Enter(mutex);
                g_autoext.Exts = null;
                g_autoext.ExtsLength = 0;
                MutexEx.Leave(mutex);
            }
        }

        public static void AutoLoadExtensions(Context ctx)
        {
            if (g_autoext.ExtsLength == 0)
                return; // Common case: early out without every having to acquire a mutex
            bool go = true;
            for (int i = 0; go; i++)
            {
                string errmsg = null;
#if THREADSAFE
                MutexEx mutex = MutexEx.Alloc(MutexEx.MUTEX.STATIC_MASTER);
#endif
                MutexEx.Enter(mutex);
                Func<Context, string, core_api_routines, RC> init;
                if (i >= g_autoext.ExtsLength)
                {
                    init = null;
                    go = false;
                }
                else
                    init = g_autoext.Exts[i];
                MutexEx.Leave(mutex);
                errmsg = null;
                RC rc;
                if (init != null && (rc = init(ctx, errmsg, g_apis)) != 0)
                {
                    Error(ctx, rc, "automatic extension loading failed: %s", errmsg);
                    go = false;
                }
                C._tagfree(ctx, ref errmsg);
            }
        }
    }
}

