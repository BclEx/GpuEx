using Core.IO;
using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
#if WINRT
using System.Threading.Tasks;
using Windows.Storage;
using Windows.Storage.Streams;
#elif WINDOWS_PHONE || SILVERLIGHT
using System.IO.IsolatedStorage;
#endif
namespace Core
{
    public class WinVSystem : VSystem
    {
        #region Preamble

#if TEST || DEBUG
        static bool OsTrace = false;
        protected static void OSTRACE(string x, params object[] args) { if (OsTrace) Console.WriteLine("b:" + string.Format(x, args)); }
#else
        protected static void OSTRACE(string x, params object[] args) { }
#endif

#if TEST
        static int io_error_hit = 0;            // Total number of I/O Errors
        static int io_error_hardhit = 0;        // Number of non-benign errors
        static int io_error_pending = 0;        // Count down to first I/O error
        static bool io_error_persist = false;   // True if I/O errors persist
        static bool io_error_benign = false;    // True if errors are benign
        static int diskfull_pending = 0;
        public static bool diskfull = false;
        protected static void SimulateIOErrorBenign(bool X) { io_error_benign = X; }
        protected static bool SimulateIOError() { if ((io_error_persist && io_error_hit > 0) || io_error_pending-- == 1) { local_ioerr(); return true; } return false; }
        protected static void local_ioerr() { OSTRACE("IOERR\n"); io_error_hit++; if (!io_error_benign) io_error_hardhit++; }
        protected static bool SimulateDiskfullError() { if (diskfull_pending > 0) { if (diskfull_pending == 1) { local_ioerr(); diskfull = true; io_error_hit = 1; return true; } else diskfull_pending--; } return false; }
#else
        protected static void SimulateIOErrorBenign(bool X) { }
        protected static bool SimulateIOError() { return false; }
        protected static bool SimulateDiskfullError() { return false; }
#endif

        // When testing, keep a count of the number of open files.
#if TEST
        static int open_file_count = 0;
        protected static void OpenCounter(int X) { open_file_count += X; }
#else
        protected static void OpenCounter(int X) { }
#endif

        #endregion

        #region Polyfill

        const int INVALID_FILE_ATTRIBUTES = -1;
        const int INVALID_SET_FILE_POINTER = -1;

#if OS_WINCE
#elif WINRT
        static bool isNT() { return true; }
#else
        static bool isNT() { return Environment.OSVersion.Platform >= PlatformID.Win32NT; }
#endif

        const long ERROR_FILE_NOT_FOUND = 2L;
        const long ERROR_HANDLE_DISK_FULL = 39L;
        const long ERROR_NOT_SUPPORTED = 50L;
        const long ERROR_DISK_FULL = 112L;

#if WINRT
        public static bool FileExists(string path)
        {
            bool exists = true;
            try { Task<StorageFile> fileTask = StorageFile.GetFileFromPathAsync(path).AsTask<StorageFile>(); fileTask.Wait(); }
            catch (Exception)
            {
                var ae = (e as AggregateException);
                if (ae != null && ae.InnerException is FileNotFoundException)
                    exists = false;
            }
            return exists;
        }
#endif

        #endregion

        #region WinVFile

        public class WinShm { }

        public partial class WinVFile : VFile
        {
            public VSystem Vfs;             // The VFS used to open this file
#if WINRT
            public IRandomAccessStream H;   // Filestream access to this file
#else
            public FileStream H;            // Filestream access to this file
#endif
            public LOCK Lock_;            // Type of lock currently held on this file
            public int SharedLockByte;      // Randomly chosen byte used as a shared lock
            public uint LastErrno;         // The Windows errno from the last I/O error
            public uint SectorSize;        // Sector size of the device file is on
#if !OMIT_WAL
            public WinShm Shm;              // Instance of shared memory on this file
#else
            public object Shm;              // DUMMY Instance of shared memory on this file
#endif
            public string Path;             // Full pathname of this file
            public int SizeChunk;           // Chunk size configured by FCNTL_CHUNK_SIZE

            public void memset()
            {
                H = null;
                Lock_ = 0;
                SharedLockByte = 0;
                LastErrno = 0;
                SectorSize = 0;
            }
        };

        #endregion

        #region OS Errors

        static RC getLastErrorMsg(ref string buf)
        {
#if SILVERLIGHT || WINRT
            buf = "Unknown error";
#else
            buf = Marshal.GetLastWin32Error().ToString();
#endif
            return RC.OK;
        }

        static RC winLogError(RC a, string b, string c)
        {
#if !WINRT
            var st = new StackTrace(new StackFrame(true)); var sf = st.GetFrame(0); return winLogErrorAtLine(a, b, c, sf.GetFileLineNumber());
#else
            return winLogErrorAtLine(a, b, c, 0);
#endif
        }
        static RC winLogErrorAtLine(RC errcode, string func, string path, int line)
        {
#if SILVERLIGHT || WINRT
            uint errno = (uint)ERROR_NOT_SUPPORTED; // Error code
#else
            uint errno = (uint)Marshal.GetLastWin32Error(); // Error code
#endif
            string msg = null; // Human readable error text
            getLastErrorMsg(ref msg);
            Debug.Assert(errcode != RC.OK);
            if (path == null) path = string.Empty;
            int i;
            for (i = 0; i < msg.Length && msg[i] != '\r' && msg[i] != '\n'; i++) { }
            msg = msg.Substring(0, i);
            SysEx.LOG(errcode, "os_win.c:%d: (%d) %s(%s) - %s", line, errno, func, path, msg);
            return errcode;
        }

        #endregion

        #region Locking

        public static bool IsRunningMediumTrust()
        {
            // this is where it needs to check if it's running in an ASP.Net MediumTrust or lower environment
            // in order to pick the appropriate locking strategy
#if SILVERLIGHT || WINRT
            return true;
#else
            return false;
#endif
        }

        private static LockingStrategy _lockingStrategy = (IsRunningMediumTrust() ? new MediumTrustLockingStrategy() : new LockingStrategy());

        /// <summary>
        /// Basic locking strategy for Console/Winform applications
        /// </summary>
        private class LockingStrategy
        {
#if !(SILVERLIGHT || WINDOWS_MOBILE || WINRT)
            [DllImport("kernel32.dll")]
            static extern bool LockFileEx(IntPtr hFile, uint dwFlags, uint dwReserved, uint nNumberOfBytesToLockLow, uint nNumberOfBytesToLockHigh, [In] ref System.Threading.NativeOverlapped lpOverlapped);
            const int LOCKFILE_FAIL_IMMEDIATELY = 1;
#endif

            public virtual void LockFile(WinVFile file, long offset, long length)
            {
#if !(SILVERLIGHT || WINDOWS_MOBILE || WINRT)
                file.H.Lock(offset, length);
#endif
            }

            public virtual int SharedLockFile(WinVFile file, long offset, long length)
            {
#if !(SILVERLIGHT || WINDOWS_MOBILE || WINRT)
                Debug.Assert(length == VFile.SHARED_SIZE);
                Debug.Assert(offset == VFile.SHARED_FIRST);
                var ovlp = new NativeOverlapped();
                ovlp.OffsetLow = (int)offset;
                ovlp.OffsetHigh = 0;
                ovlp.EventHandle = IntPtr.Zero;
                //SafeFileHandle.DangerousGetHandle().ToInt32()
                return (LockFileEx(file.H.Handle, LOCKFILE_FAIL_IMMEDIATELY, 0, (uint)length, 0, ref ovlp) ? 1 : 0);
#else
            return 1;
#endif
            }

            public virtual void UnlockFile(WinVFile file, long offset, long length)
            {
#if !(SILVERLIGHT || WINDOWS_MOBILE || WINRT)
                file.H.Unlock(offset, length);
#endif
            }
        }

        /// <summary>
        /// Locking strategy for Medium Trust. It uses the same trick used in the native code for WIN_CE
        /// which doesn't support LockFileEx as well.
        /// </summary>
        private class MediumTrustLockingStrategy : LockingStrategy
        {
            public override int SharedLockFile(WinVFile file, long offset, long length)
            {
#if !(SILVERLIGHT || WINDOWS_MOBILE || WINRT)
                Debug.Assert(length == VFile.SHARED_SIZE);
                Debug.Assert(offset == VFile.SHARED_FIRST);
                try { file.H.Lock(offset + file.SharedLockByte, 1); }
                catch (IOException) { return 0; }
#endif
                return 1;
            }
        }

        #endregion

        #region WinVFile

        public partial class WinVFile : VFile
        {
            static int seekWinFile(WinVFile file, long offset)
            {
                try
                {
#if WINRT
                    file.H.Seek((ulong)offset); 
#else
                    file.H.Seek(offset, SeekOrigin.Begin);
#endif
                }
                catch (Exception)
                {
#if SILVERLIGHT || WINRT
                    file.LastErrno = 1;
#else
                    file.LastErrno = (uint)Marshal.GetLastWin32Error();
#endif
                    winLogError(RC.IOERR_SEEK, "seekWinFile", file.Path);
                    return 1;
                }
                return 0;
            }

            public static int MX_CLOSE_ATTEMPT = 3;
            public override RC Close()
            {
#if !OMIT_WAL
                Debug.Assert(Shm == null);
#endif
                OSTRACE("CLOSE %d (%s)\n", H.GetHashCode(), H.Name);
                bool rc;
                int cnt = 0;
                do
                {
#if WINRT
                    H.Dispose();
#else
                    H.Close();
#endif
                    rc = true;
                } while (!rc && ++cnt < MX_CLOSE_ATTEMPT);
                OSTRACE("CLOSE %d %s\n", H.GetHashCode(), rc ? "ok" : "failed");
                if (rc)
                    H = null;
                OpenCounter(-1);
                return (rc ? RC.OK : winLogError(RC.IOERR_CLOSE, "winClose", Path));
            }

            public override RC Read(byte[] buffer, int amount, long offset)
            {
                //if (buffer == null)
                //    buffer = new byte[amount];
                if (SimulateIOError())
                    return RC.IOERR_READ;
                OSTRACE("READ %d lock=%d\n", H.GetHashCode(), Lock_);
                if (!H.CanRead)
                    return RC.IOERR_READ;
                if (seekWinFile(this, offset) != 0)
                    return RC.FULL;
                int read; // Number of bytes actually read from file
                try
                {
#if WINRT
                    var stream = H.AsStreamForRead();
                    read = stream.Read(buffer, 0, amount);
#else
                    read = H.Read(buffer, 0, amount);
#endif
                }
                catch (Exception)
                {
#if SILVERLIGHT || WINRT
                    LastErrno = 1;
#else
                    LastErrno = (uint)Marshal.GetLastWin32Error();
#endif
                    return winLogError(RC.IOERR_READ, "winRead", Path);
                }
                if (read < amount)
                {
                    // Unread parts of the buffer must be zero-filled
                    Array.Clear(buffer, (int)read, (int)(amount - read));
                    return RC.IOERR_SHORT_READ;
                }
                return RC.OK;
            }

            public override RC Write(byte[] buffer, int amount, long offset)
            {
                Debug.Assert(amount > 0);
                if (SimulateIOError())
                    return RC.IOERR_WRITE;
                if (SimulateDiskfullError())
                    return RC.FULL;
                OSTRACE("WRITE %d lock=%d\n", H.GetHashCode(), Lock_);
                int rc = seekWinFile(this, offset); // True if error has occured, else false
#if WINRT
                ulong wrote = H.Position;
#else
                long wrote = H.Position;
#endif
                try
                {
                    Debug.Assert(buffer.Length >= amount);
#if WINRT
                    var stream = H.AsStreamForWrite();
                    stream.Write(buffer, 0, amount);
#else
                    H.Write(buffer, 0, amount);
#endif
                    rc = 1;
                    wrote = H.Position - wrote;
                }
                catch (IOException) { return RC.READONLY; }
                if (rc == 0 || amount > (int)wrote)
                {
#if SILVERLIGHT || WINRT
                    LastErrno  = 1;
#else
                    LastErrno = (uint)Marshal.GetLastWin32Error();
#endif
                    if (LastErrno == ERROR_HANDLE_DISK_FULL || LastErrno == ERROR_DISK_FULL)
                        return RC.FULL;
                    else
                        return winLogError(RC.IOERR_WRITE, "winWrite", Path);
                }
                return RC.OK;
            }

            public override RC Truncate(long size)
            {
                RC rc = RC.OK;
                OSTRACE("TRUNCATE %d %lld\n", H.Name, size);
                if (SimulateIOError())
                    return RC.IOERR_TRUNCATE;
                // If the user has configured a chunk-size for this file, truncate the file so that it consists of an integer number of chunks (i.e. the
                // actual file size after the operation may be larger than the requested size).
                if (SizeChunk > 0)
                    size = ((size + SizeChunk - 1) / SizeChunk) * SizeChunk;
                try
                {
#if WINRT
                    H.Size = (ulong)size;
#else
                    H.SetLength(size);
#endif
                    rc = RC.OK;
                }
                catch (IOException)
                {
#if SILVERLIGHT || WINRT
                    LastErrno = 1;
#else
                    LastErrno = (uint)Marshal.GetLastWin32Error();
#endif
                    rc = winLogError(RC.IOERR_TRUNCATE, "winTruncate2", Path);
                }
                OSTRACE("TRUNCATE %d %lld %s\n", H.GetHashCode(), size, rc == RC.OK ? "ok" : "failed");
                return rc;
            }

#if TEST
            // Count the number of fullsyncs and normal syncs.  This is used to test that syncs and fullsyncs are occuring at the right times.
#if !TCLSH
            static int sync_count = 0;
            static int fullsync_count = 0;
#else
            static tcl.lang.Var.SQLITE3_GETSET sync_count = new tcl.lang.Var.SQLITE3_GETSET("sync_count");
            static tcl.lang.Var.SQLITE3_GETSET fullsync_count = new tcl.lang.Var.SQLITE3_GETSET("fullsync_count");
#endif
#endif

            public override RC Sync(SYNC flags)
            {
                // Check that one of SQLITE_SYNC_NORMAL or FULL was passed
                Debug.Assert(((int)flags & 0x0F) == (int)SYNC.NORMAL || ((int)flags & 0x0F) == (int)SYNC.FULL);
                OSTRACE("SYNC %d lock=%d\n", H.GetHashCode(), Lock_);
                // Unix cannot, but some systems may return SQLITE_FULL from here. This line is to test that doing so does not cause any problems.
                if (SimulateDiskfullError())
                    return RC.FULL;
#if TEST
                if (((int)flags & 0x0F) == (int)SYNC.FULL)
#if !TCLSH
                    fullsync_count++;
                sync_count++;
#else
                    fullsync_count.iValue++;
                sync_count.iValue++;
#endif
#endif
#if NO_SYNC // If we compiled with the SQLITE_NO_SYNC flag, then syncing is a no-op
                return RC_OK;
#elif WINRT
                var stream = H.AsStreamForWrite();
                stream.Flush();
                return RC.OK;
#else
                H.Flush();
                return RC.OK;
#endif
            }

            public override RC get_FileSize(out long size)
            {
                if (SimulateIOError())
                {
                    size = 0;
                    return RC.IOERR_FSTAT;
                }
#if WINRT
                size = (H.CanRead ? (long)H.Size : 0);
#else
                size = (H.CanRead ? H.Length : 0);
#endif
                return RC.OK;
            }

            static int getReadLock(WinVFile file)
            {
                int res = 0;
                if (isNT())
                    res = _lockingStrategy.SharedLockFile(file, SHARED_FIRST, SHARED_SIZE);
                // isNT() is 1 if SQLITE_OS_WINCE==1, so this else is never executed.
#if !OS_WINCE
                else
                {
                    Debugger.Break();
                    //  int lk;
                    //  sqlite3_randomness(lk.Length, lk);
                    //  pFile.sharedLockByte = (u16)((lk & 0x7fffffff)%(SHARED_SIZE - 1));
                    //  res = pFile.fs.Lock( SHARED_FIRST + pFile.sharedLockByte, 0, 1, 0);
                }
#endif
                if (res == 0)
#if SILVERLIGHT || WINRT
                    file.LastErrno = 1;
#else
                    file.LastErrno = (uint)Marshal.GetLastWin32Error();
#endif
                // No need to log a failure to lock
                return res;
            }

            static int unlockReadLock(WinVFile file)
            {
                int res = 1;
                if (isNT())
                    try { _lockingStrategy.UnlockFile(file, SHARED_FIRST, SHARED_SIZE); }
                    catch (Exception) { res = 0; }
                // isNT() is 1 if SQLITE_OS_WINCE==1, so this else is never executed.
#if !OS_WINCE
                else
                    Debugger.Break();
#endif
                if (res == 0)
                {
#if SILVERLIGHT || WINRT
                    file.LastErrno = 1;
#else
                    file.LastErrno = (uint)Marshal.GetLastWin32Error();
#endif
                    winLogError(RC.IOERR_UNLOCK, "unlockReadLock", file.Path);
                }
                return res;
            }

            public override RC Lock(LOCK lock_)
            {
                OSTRACE("LOCK %d %d was %d(%d)\n", H.GetHashCode(), lock_, Lock_, SharedLockByte);

                // If there is already a lock of this type or more restrictive on the OsFile, do nothing. Don't use the end_lock: exit path, as
                // sqlite3OsEnterMutex() hasn't been called yet.
                if (Lock_ >= lock_)
                    return RC.OK;

                // Make sure the locking sequence is correct
                Debug.Assert(lock_ != LOCK.NO || lock_ == LOCK.SHARED);
                Debug.Assert(lock_ != LOCK.PENDING);
                Debug.Assert(lock_ != LOCK.RESERVED || Lock_ == LOCK.SHARED);

                // Lock the PENDING_LOCK byte if we need to acquire a PENDING lock or a SHARED lock.  If we are acquiring a SHARED lock, the acquisition of
                // the PENDING_LOCK byte is temporary.
                LOCK newLock = Lock_; // Set pFile.locktype to this value before exiting
                int res = 1;                // Result of a windows lock call
                bool gotPendingLock = false;// True if we acquired a PENDING lock this time
                uint lastErrno = 0;
                if (Lock_ == LOCK.NO || (lock_ == LOCK.EXCLUSIVE && Lock_ == LOCK.RESERVED))
                {
                    res = 0;
                    int cnt = 3;
                    while (cnt-- > 0 && res == 0)
                    {
                        try { _lockingStrategy.LockFile(this, PENDING_BYTE, 1); res = 1; }
                        catch (Exception)
                        {
                            // Try 3 times to get the pending lock.  The pending lock might be held by another reader process who will release it momentarily.
                            OSTRACE("could not get a PENDING lock. cnt=%d\n", cnt);
#if WINRT
                            System.Threading.Tasks.Task.Delay(1).Wait();
#else
                            Thread.Sleep(1);
#endif
                        }
                    }
                    gotPendingLock = (res != 0);
                    if (res == 0)
#if SILVERLIGHT || WINRT
                        lastErrno = 1;
#else
                        lastErrno = (uint)Marshal.GetLastWin32Error();
#endif
                }

                // Acquire a SHARED lock
                if (lock_ == LOCK.SHARED && res != 0)
                {
                    Debug.Assert(Lock_ == LOCK.NO);
                    res = getReadLock(this);
                    if (res != 0)
                        newLock = LOCK.SHARED;
                    else
#if SILVERLIGHT || WINRT
                        lastErrno = 1;
#else
                        lastErrno = (uint)Marshal.GetLastWin32Error();
#endif
                }

                // Acquire a RESERVED lock
                if (lock_ == LOCK.RESERVED && res != 0)
                {
                    Debug.Assert(Lock_ == LOCK.SHARED);
                    try { _lockingStrategy.LockFile(this, RESERVED_BYTE, 1); newLock = LOCK.RESERVED; res = 1; }
                    catch (Exception) { res = 0; }
                    if (res != 0)
                        newLock = LOCK.RESERVED;
                    else
#if SILVERLIGHT
                        lastErrno = 1;
#else
                        lastErrno = (uint)Marshal.GetLastWin32Error();
#endif
                }

                // Acquire a PENDING lock
                if (lock_ == LOCK.EXCLUSIVE && res != 0)
                {
                    newLock = LOCK.PENDING;
                    gotPendingLock = false;
                }

                // Acquire an EXCLUSIVE lock
                if (lock_ == LOCK.EXCLUSIVE && res != 0)
                {
                    Debug.Assert(Lock_ >= LOCK.SHARED);
                    res = unlockReadLock(this);
                    OSTRACE("unreadlock = %d\n", res);
                    try { _lockingStrategy.LockFile(this, SHARED_FIRST, SHARED_SIZE); newLock = LOCK.EXCLUSIVE; res = 1; }
                    catch (Exception) { res = 0; }
                    if (res != 0)
                        newLock = LOCK.EXCLUSIVE;
                    else
                    {
#if SILVERLIGHT || WINRT
                        lastErrno = 1;
#else
                        lastErrno = (uint)Marshal.GetLastWin32Error();
#endif
                        OSTRACE("error-code = %d\n", lastErrno);
                        getReadLock(this);
                    }
                }

                // If we are holding a PENDING lock that ought to be released, then release it now.
                if (gotPendingLock && lock_ == LOCK.SHARED)
                    _lockingStrategy.UnlockFile(this, PENDING_BYTE, 1);

                // Update the state of the lock has held in the file descriptor then return the appropriate result code.
                RC rc;
                if (res != 0)
                    rc = RC.OK;
                else
                {
                    OSTRACE("LOCK FAILED %d trying for %d but got %d\n", H.GetHashCode(), lock_, newLock);
                    LastErrno = lastErrno;
                    rc = RC.BUSY;
                }
                Lock_ = newLock;
                return rc;
            }

            public override RC CheckReservedLock(ref int resOut)
            {
                if (SimulateIOError())
                    return RC.IOERR_CHECKRESERVEDLOCK;
                int rc;
                if (Lock_ >= LOCK.RESERVED)
                {
                    rc = 1;
                    OSTRACE("TEST WR-LOCK %d %d (local)\n", H.Name, rc);
                }
                else
                {
                    try { _lockingStrategy.LockFile(this, RESERVED_BYTE, 1); _lockingStrategy.UnlockFile(this, RESERVED_BYTE, 1); rc = 1; }
                    catch (IOException) { rc = 0; }
                    rc = 1 - rc;
                    OSTRACE("TEST WR-LOCK %d %d (remote)\n", H.GetHashCode(), rc);
                }
                resOut = rc;
                return RC.OK;
            }

            public override RC Unlock(LOCK lock_)
            {
                Debug.Assert(lock_ <= LOCK.SHARED);
                OSTRACE("UNLOCK %d to %d was %d(%d)\n", H.GetHashCode(), lock_, Lock_, SharedLockByte);
                var rc = RC.OK;
                LOCK type = Lock_;
                if (type >= LOCK.EXCLUSIVE)
                {
                    _lockingStrategy.UnlockFile(this, SHARED_FIRST, SHARED_SIZE);
                    if (lock_ == LOCK.SHARED && getReadLock(this) == 0) // This should never happen.  We should always be able to reacquire the read lock
                        rc = winLogError(RC.IOERR_UNLOCK, "winUnlock", Path);
                }
                if (type >= LOCK.RESERVED)
                    try { _lockingStrategy.UnlockFile(this, RESERVED_BYTE, 1); }
                    catch (Exception) { }
                if (lock_ == LOCK.NO && type >= LOCK.SHARED)
                    unlockReadLock(this);
                if (type >= LOCK.PENDING)
                    try { _lockingStrategy.UnlockFile(this, PENDING_BYTE, 1); }
                    catch (Exception) { }
                Lock_ = lock_;
                return rc;
            }

            //static void winModeBit(WinVFile file, char mask, ref long arg)
            //{
            //    if (arg < 0)
            //        arg = ((file.CtrlFlags & mask) != 0);
            //    else if (arg == 0)
            //        file.CtrlFlags &= ~mask;
            //    else
            //        file.CtrlFlags |= mask;
            //}

            public override RC FileControl(FCNTL op, ref long arg)
            {
                switch (op)
                {
                    case FCNTL.LOCKSTATE:
                        arg = (int)Lock_;
                        return RC.OK;
                    case FCNTL.LAST_ERRNO:
                        arg = (int)LastErrno;
                        return RC.OK;
                    case FCNTL.CHUNK_SIZE:
                        SizeChunk = (int)arg;
                        return RC.OK;
                    case FCNTL.SIZE_HINT:
                        if (SizeChunk > 0)
                        {
                            long oldSize;
                            var rc = get_FileSize(out oldSize);
                            if (rc == RC.OK)
                            {
                                var newSize = (long)arg;
                                if (newSize > oldSize)
                                {
                                    SimulateIOErrorBenign(true);
                                    Truncate(newSize);
                                    SimulateIOErrorBenign(false);
                                }
                            }
                            return rc;
                        }
                        return RC.OK;
                    case FCNTL.PERSIST_WAL:
                        //winModeBit(this, WINFILE_PERSIST_WAL, ref arg);
                        return RC.OK;
                    case FCNTL.POWERSAFE_OVERWRITE:
                        //winModeBit(this, WINFILE_PSOW, ref arg);
                        return RC.OK;
                    //case FCNTL.VFSNAME:
                    //    arg = "win32";
                    //    return RC.OK;
                    //case FCNTL.WIN32_AV_RETRY:
                    //    int *a = (int*)arg;
                    //    if (a[0] > 0)
                    //        win32IoerrRetry = a[0];
                    //    else
                    //        a[0] = win32IoerrRetry;
                    //    if (a[1] > 0)
                    //        win32IoerrRetryDelay = a[1];
                    //    else
                    //        a[1] = win32IoerrRetryDelay;
                    //    return RC.OK;
                    //case FCNTL.TEMPFILENAME:
                    //    var tfile = _alloc(Vfs->MaxPathname, true);
                    //    if (tfile)
                    //    {
                    //        getTempname(Vfs->MaxPathname, tfile);
                    //        *(char**)arg = tfile;
                    //    }
                    //    return RC.OK;
                }
                return RC.NOTFOUND;
            }

            public override uint get_SectorSize()
            {
                //return DEFAULT_SECTOR_SIZE;
                return SectorSize;
            }

            //public override IOCAP get_DeviceCharacteristics() { return 0; }

#if !OMIT_WAL
            public override RC ShmMap(int region, int sizeRegion, bool isWrite, out object pp) { pp = null; return RC.OK; }
            public override RC ShmLock(int offset, int count, SHM flags) { return RC.OK; }
            public override void ShmBarrier() { }
            public override RC ShmUnmap(bool deleteFlag) { return RC.OK; }
#endif

        }

        #endregion

        #region WinVSystem

        //static string ConvertUtf8Filename(string filename)
        //{
        //    return filename;
        //}
        //        static RC getTempname(int bufLength, StringBuilder buf)
        //        {
        //            const string chars = "abcdefghijklmnopqrstuvwxyz0123456789";
        //            var random = new StringBuilder(20);
        //            long randomValue = 0;
        //            for (int i = 0; i < 15; i++)
        //            {
        //                sqlite3_randomness(1, ref randomValue);
        //                random.Append((char)chars[(int)(randomValue % (chars.Length - 1))]);
        //            }
        //#if WINRT
        //                    buf.Append(Path.Combine(ApplicationData.Current.LocalFolder.Path, TEMP_FILE_PREFIX + random.ToString()));
        //#else
        //            buf.Append(Path.GetTempPath() + TEMP_FILE_PREFIX + random.ToString());
        //#endif
        //            OSTRACE("TEMP FILENAME: %s\n", buf.ToString());
        //            return RC.OK;
        //        }

        public override RC Open(string name, VFile id, OPEN flags, out OPEN outFlags)
        {
            // 0x87f7f is a mask of SQLITE_OPEN_ flags that are valid to be passed down into the VFS layer.  Some SQLITE_OPEN_ flags (for example,
            // SQLITE_OPEN_FULLMUTEX or SQLITE_OPEN_SHAREDCACHE) are blocked before reaching the VFS.
            flags = (OPEN)((uint)flags & 0x87f7f);
            outFlags = 0;

            var rc = RC.OK;
            var type = (OPEN)(int)((int)flags & 0xFFFFFF00); // Type of file to open
            var isExclusive = (flags & OPEN.EXCLUSIVE) != 0;
            var isDelete = (flags & OPEN.DELETEONCLOSE) != 0;
            var isCreate = (flags & OPEN.CREATE) != 0;
            var isReadonly = (flags & OPEN.READONLY) != 0;
            var isReadWrite = (flags & OPEN.READWRITE) != 0;
            var isOpenJournal = (isCreate && (type == OPEN.MASTER_JOURNAL || type == OPEN.MAIN_JOURNAL || type == OPEN.WAL));

            // Check the following statements are true:
            //
            //   (a) Exactly one of the READWRITE and READONLY flags must be set, and
            //   (b) if CREATE is set, then READWRITE must also be set, and
            //   (c) if EXCLUSIVE is set, then CREATE must also be set.
            //   (d) if DELETEONCLOSE is set, then CREATE must also be set.
            Debug.Assert((!isReadonly || !isReadWrite) && (isReadWrite || isReadonly));
            Debug.Assert(!isCreate || isReadWrite);
            Debug.Assert(!isExclusive || isCreate);
            Debug.Assert(!isDelete || isCreate);

            // The main DB, main journal, WAL file and master journal are never automatically deleted. Nor are they ever temporary files.
            //Debug.Assert((!isDelete && !string.IsNullOrEmpty(name)) || type != OPEN.MAIN_DB);
            Debug.Assert((!isDelete && !string.IsNullOrEmpty(name)) || type != OPEN.MAIN_JOURNAL);
            Debug.Assert((!isDelete && !string.IsNullOrEmpty(name)) || type != OPEN.MASTER_JOURNAL);
            Debug.Assert((!isDelete && !string.IsNullOrEmpty(name)) || type != OPEN.WAL);

            // Assert that the upper layer has set one of the "file-type" flags.
            Debug.Assert(type == OPEN.MAIN_DB || type == OPEN.TEMP_DB ||
                type == OPEN.MAIN_JOURNAL || type == OPEN.TEMP_JOURNAL ||
                type == OPEN.SUBJOURNAL || type == OPEN.MASTER_JOURNAL ||
                type == OPEN.TRANSIENT_DB || type == OPEN.WAL);

            var file = (WinVFile)id;
            Debug.Assert(file != null);
            file.H = null;

            // If the second argument to this function is NULL, generate a temporary file name to use
            if (string.IsNullOrEmpty(name))
            {
                Debug.Assert(isDelete && !isOpenJournal);
                name = Path.GetRandomFileName();
            }

            // Convert the filename to the system encoding.
            if (name.StartsWith("/") && !name.StartsWith("//"))
                name = name.Substring(1);
#if !WINRT
            FileAccess dwDesiredAccess;
            if (isReadWrite)
                dwDesiredAccess = FileAccess.Read | FileAccess.Write;
            else
                dwDesiredAccess = FileAccess.Read;

            // SQLITE_OPEN_EXCLUSIVE is used to make sure that a new file is created. SQLite doesn't use it to indicate "exclusive access"
            // as it is usually understood.
            FileMode dwCreationDisposition;
            if (isExclusive) // Creates a new file, only if it does not already exist. If the file exists, it fails.
                dwCreationDisposition = FileMode.CreateNew;
            else if (isCreate) // Open existing file, or create if it doesn't exist
                dwCreationDisposition = FileMode.OpenOrCreate;
            else // Opens a file, only if it exists.
                dwCreationDisposition = FileMode.Open;
            FileShare dwShareMode = FileShare.Read | FileShare.Write;
#endif

#if OS_WINCE
            uint dwDesiredAccess = 0;
            int isTemp = 0;
#else
#if !(SILVERLIGHT || WINDOWS_MOBILE || WINRT)
            FileOptions dwFlagsAndAttributes;
#endif
#endif
            if (isDelete)
            {
#if OS_WINCE
                dwFlagsAndAttributes = FILE_ATTRIBUTE_HIDDEN;
                isTemp = 1;
#else
#if !(SILVERLIGHT || WINDOWS_MOBILE || WINRT)
                dwFlagsAndAttributes = FileOptions.DeleteOnClose;
#endif
#endif
            }
            else
            {
#if !(SILVERLIGHT || WINDOWS_MOBILE || SQLITE_WINRT)
                dwFlagsAndAttributes = FileOptions.None;
#endif
            }
            // Reports from the internet are that performance is always better if FILE_FLAG_RANDOM_ACCESS is used.  Ticket #2699.
#if OS_WINCE
            dwFlagsAndAttributes |= FileOptions.RandomAccess;
#endif

#if WINRT
            IRandomAccessStream fs = null;
            DWORD dwDesiredAccess = 0;
#else
            FileStream fs = null;
#endif
            if (isNT())
            {
                // retry opening the file a few times; this is because of a racing condition between a delete and open call to the FS
                int retries = 3;
                while (fs == null && retries > 0)
                    try
                    {
                        retries--;
#if WINRT
                        Task<StorageFile> fileTask = null;
                        if (isExclusive)
                        {
                            if (HelperMethods.FileExists(name)) // Error
                                
                                throw new IOException("file already exists");
                            else
                            {
                                Task<StorageFolder> folderTask = StorageFolder.GetFolderFromPathAsync(Path.GetDirectoryName(name)).AsTask<StorageFolder>();
                                folderTask.Wait();
                                fileTask = folderTask.Result.CreateFileAsync(Path.GetFileName(name)).AsTask<StorageFile>();
                            }
                        }
                        else if (isCreate)
                        {
                            if (HelperMethods.FileExists(name))
                                fileTask = StorageFile.GetFileFromPathAsync(name).AsTask<StorageFile>();
                            else
                            {
                                Task<StorageFolder> folderTask = StorageFolder.GetFolderFromPathAsync(Path.GetDirectoryName(name)).AsTask<StorageFolder>();
                                folderTask.Wait();
                                fileTask = folderTask.Result.CreateFileAsync(Path.GetFileName(name)).AsTask<StorageFile>();
                            }
                        }
                        else
                            fileTask = StorageFile.GetFileFromPathAsync(name).AsTask<StorageFile>();
                        fileTask.Wait();
                        Task<IRandomAccessStream> streamTask = fileTask.Result.OpenAsync(FileAccessMode.ReadWriteUnsafe).AsTask<IRandomAccessStream>();
                        streamTask.Wait();
                        fs = streamTask.Result;
#elif WINDOWS_PHONE || SILVERLIGHT  
                        fs = new IsolatedStorageFileStream(name, dwCreationDisposition, dwDesiredAccess, dwShareMode, IsolatedStorageFile.GetUserStoreForApplication());
#elif !(SQLITE_SILVERLIGHT || WINDOWS_MOBILE)
                        fs = new FileStream(name, dwCreationDisposition, dwDesiredAccess, dwShareMode, 4096, dwFlagsAndAttributes);
#else
                        fs = new FileStream(name, dwCreationDisposition, dwDesiredAccess, dwShareMode, 4096);
#endif
                        OSTRACE("OPEN %d (%s)\n", fs.GetHashCode(), fs.Name);
                    }
                    catch (Exception)
                    {
#if WINRT
                        System.Threading.Tasks.Task.Delay(100).Wait();

#else
                        Thread.Sleep(100);
#endif
                    }

                // isNT() is 1 if SQLITE_OS_WINCE==1, so this else is never executed. Since the ASCII version of these Windows API do not exist for WINCE,
                // it's important to not reference them for WINCE builds.
#if !OS_WINCE
            }
            else
            {
                Debugger.Break();
#endif
            }

            OSTRACE("OPEN {0} {1} 0x{2:x} {3}\n", file.GetHashCode(), name, dwDesiredAccess, fs == null ? "failed" : "ok");
            if (fs == null ||
#if !(SQLITE_SILVERLIGHT || WINDOWS_MOBILE || SQLITE_WINRT)
 fs.SafeFileHandle.IsInvalid
#else
!fs.CanRead
#endif
)
            {
#if SILVERLIGHT || WINRT
                file.LastErrno = 1;
#else
                file.LastErrno = (uint)Marshal.GetLastWin32Error();
#endif
                winLogError(RC.CANTOPEN, "winOpen", name);
                if (isReadWrite)
                    return Open(name, file, ((flags | OPEN.READONLY) & ~(OPEN.CREATE | OPEN.READWRITE)), out outFlags);
                else
                    return SysEx.CANTOPEN_BKPT();
            }
            outFlags = (isReadWrite ? OPEN.READWRITE : OPEN.READONLY);
            file.memset();
            file.Opened = true;
            file.H = fs;
            file.LastErrno = 0;
            file.Vfs = this;
            file.Shm = null;
            file.Path = name;
            file.SectorSize = (uint)getSectorSize(this, name);
#if OS_WINCE
            if (isReadWrite && type == OPEN.MAIN_DB && !winceCreateLock(name, file))
            {
                CloseHandle(h);
                return SysEx.CANTOPEN_BKPT();
            }
            if (isTemp)
                file.DeleteOnClose = name;
#endif
            OpenCounter(+1);
            return rc;
        }

        static int MX_DELETION_ATTEMPTS = 5;
        public override RC Delete(string filename, bool syncDir)
        {
            if (SimulateIOError())
                return RC.IOERR_DELETE;

            int cnt = 0;
            RC rc = RC.ERROR;


            if (isNT())
                do
                {
#if WINRT
            if(!HelperMethods.FileExists(filename))
#elif WINDOWS_PHONE
           if (!System.IO.IsolatedStorage.IsolatedStorageFile.GetUserStoreForApplication().FileExists(filename))
#elif SILVERLIGHT
            if (!IsolatedStorageFile.GetUserStoreForApplication().FileExists(filename))
#else
                    if (!File.Exists(filename))
#endif
                    {
                        rc = RC.IOERR;
                        break;
                    }
                    try
                    {
#if WINRT
              Task<StorageFile> fileTask = StorageFile.GetFileFromPathAsync(filename).AsTask<StorageFile>();
              fileTask.Wait();
              fileTask.Result.DeleteAsync().AsTask().Wait();
#elif WINDOWS_PHONE
              System.IO.IsolatedStorage.IsolatedStorageFile.GetUserStoreForApplication().DeleteFile(filename);
#elif SILVERLIGHT
              IsolatedStorageFile.GetUserStoreForApplication().DeleteFile(filename);
#else
                        File.Delete(filename);
#endif
                        rc = RC.OK;
                    }
                    catch (IOException)
                    {
                        rc = RC.IOERR;
#if WINRT
                        System.Threading.Tasks.Task.Delay(100).Wait();
#else
                        Thread.Sleep(100);
#endif
                    }
                } while (rc != RC.OK && ++cnt < MX_DELETION_ATTEMPTS);
            // isNT() is 1 if SQLITE_OS_WINCE==1, so this else is never executed. Since the ASCII version of these Windows API do not exist for WINCE,
            // it's important to not reference them for WINCE builds.
#if !OS_WINCE && !WINRT
            else
                do
                {
                    if (!File.Exists(filename))
                    {
                        rc = RC.IOERR;
                        break;
                    }
                    try
                    {
                        File.Delete(filename);
                        rc = RC.OK;
                    }
                    catch (IOException)
                    {
                        rc = RC.IOERR;
                        Thread.Sleep(100);
                    }
                } while (rc != RC.OK && cnt++ < MX_DELETION_ATTEMPTS);
#endif
            OSTRACE("DELETE \"%s\"\n", filename);
            if (rc == RC.OK)
                return rc;

            int lastErrno;
#if SILVERLIGHT || WINRT
            lastErrno = (int)ERROR_NOT_SUPPORTED;
#else
            lastErrno = Marshal.GetLastWin32Error();
#endif
            return (lastErrno == ERROR_FILE_NOT_FOUND ? RC.OK : winLogError(RC.IOERR_DELETE, "winDelete", filename));
        }

        public override RC Access(string filename, ACCESS flags, out int resOut)
        {
            if (SimulateIOError())
            {
                resOut = -1;
                return RC.IOERR_ACCESS;
            }
            // Do a quick test to prevent the try/catch block
            if (flags == ACCESS.EXISTS)
            {
#if WINRT
                resOut = HelperMethods.FileExists(zFilename) ? 1 : 0;
#elif WINDOWS_PHONE
                resOut = System.IO.IsolatedStorage.IsolatedStorageFile.GetUserStoreForApplication().FileExists(zFilename) ? 1 : 0;
#elif SILVERLIGHT
                resOut = IsolatedStorageFile.GetUserStoreForApplication().FileExists(zFilename) ? 1 : 0;
#else
                resOut = File.Exists(filename) ? 1 : 0;
#endif
                return RC.OK;
            }
            FileAttributes attr = 0;
            try
            {
#if WINRT
                 attr = FileAttributes.Normal;
            }
#else
#if WINDOWS_PHONE || WINDOWS_MOBILE || SILVERLIGHT
            if (new DirectoryInfo(filename).Exists)
#else
                attr = File.GetAttributes(filename);
                if (attr == FileAttributes.Directory)
#endif
                {
                    try
                    {
                        var name = Path.Combine(Path.GetTempPath(), Path.GetTempFileName());
                        var fs = File.Create(name);
                        fs.Close();
                        File.Delete(name);
                        attr = FileAttributes.Normal;
                    }
                    catch (IOException) { attr = FileAttributes.ReadOnly; }
                }
            }
            // isNT() is 1 if SQLITE_OS_WINCE==1, so this else is never executed. Since the ASCII version of these Windows API do not exist for WINCE,
            // it's important to not reference them for WINCE builds.
#if !OS_WINCE
#endif
#endif
            catch (IOException) { winLogError(RC.IOERR_ACCESS, "winAccess", filename); }
            int rc = 0;
            switch (flags)
            {
                case ACCESS.READ:
                case ACCESS.EXISTS:
                    rc = attr != 0 ? 1 : 0;
                    break;
                case ACCESS.READWRITE:
                    rc = attr == 0 ? 0 : (int)(attr & FileAttributes.ReadOnly) != 0 ? 0 : 1;
                    break;
                default:
                    Debug.Assert("Invalid flags argument" == "");
                    rc = 0;
                    break;
            }
            resOut = rc;
            return RC.OK;
        }

        public override RC FullPathname(string relative, out string full)
        {
#if OS_WINCE
            if (SimulateIOError())
                return RC.ERROR;
            // WinCE has no concept of a relative pathname, or so I am told.
            snprintf(MaxPathname, full, "%s", relative);
            return RC.OK;
#endif
#if !OS_WINCE
            full = null;
            // If this path name begins with "/X:", where "X" is any alphabetic character, discard the initial "/" from the pathname.
            if (relative[0] == '/' && Char.IsLetter(relative[1]) && relative[2] == ':')
                relative = relative.Substring(1);
            if (SimulateIOError())
                return RC.ERROR;
            if (isNT())
            {
                try
                {
#if WINDOWS_PHONE || SILVERLIGHT  || WINRT
                    full = relative;
#else
                    full = Path.GetFullPath(relative);
#endif
                }
                catch (Exception) { full = relative; }
#if !SQLITE_OS_WINCE
            }
            else
            {
                Debugger.Break();
#endif
            }
            if (full.Length > MaxPathname)
                full = full.Substring(0, MaxPathname);
            return RC.OK;
#endif
        }


        const int DEFAULT_SECTOR_SIZE = 512;
        static int getSectorSize(VSystem vfs, string relative)
        {
            return DEFAULT_SECTOR_SIZE;
        }

#if !OMIT_LOAD_EXTENSION
        public override object DlOpen(string filename) { throw new NotSupportedException(); }
        public override void DlError(int bufLength, string buf) { throw new NotSupportedException(); }
        public override object DlSym(object handle, string symbol) { throw new NotSupportedException(); }
        public override void DlClose(object handle) { throw new NotSupportedException(); }
#else
        public override object DlOpen(string filename) { return null; }
        public override void DlError(int byteLength, string errMsg) { return 0; }
        public override object DlSym(object data, string symbol) { return null; }
        public override void DlClose(object data) { return 0; }
#endif

        public override int Randomness(int bufLength, byte[] buf)
        {
            int n = 0;
#if TEST
            n = bufLength;
            Array.Clear(buf, 0, n);
#else
            var sBuf = BitConverter.GetBytes(DateTime.Now.Ticks);
            buf[0] = sBuf[0];
            buf[1] = sBuf[1];
            buf[2] = sBuf[2];
            buf[3] = sBuf[3];
            n += 16;
            if (sizeof(uint) <= bufLength - n)
            {
                uint processId;
#if !(SILVERLIGHT || WINRT)
                processId = (uint)Process.GetCurrentProcess().Id;
#else
                processId = 28376023;
#endif
                ConvertEx.Put4(buf, n, processId);
                n += 4;
            }
            if (sizeof(uint) <= bufLength - n)
            {
                var dt = new DateTime();
                ConvertEx.Put4(buf, n, (uint)dt.Ticks);// memcpy(&zBuf[n], cnt, sizeof(cnt));
                n += 4;
            }
            if (sizeof(long) <= bufLength - n)
            {
                long i;
                i = DateTime.UtcNow.Millisecond;
                ConvertEx.Put4(buf, n, (uint)(i & 0xFFFFFFFF));
                ConvertEx.Put4(buf, n, (uint)(i >> 32));
                n += sizeof(long);
            }
#endif
            return n;
        }

        public override int Sleep(int microsec)
        {
#if WINRT
            System.Threading.Tasks.Task.Delay(((microsec + 999) / 1000)).Wait();

#else
            Thread.Sleep(((microsec + 999) / 1000));
#endif
            return ((microsec + 999) / 1000) * 1000;
        }

#if TEST
#if !TCLSH
        static int current_time = 0;  // Fake system time in seconds since 1970.
#else
    static tcl.lang.Var.SQLITE3_GETSET current_time = new tcl.lang.Var.SQLITE3_GETSET("current_time");
#endif
#endif

        public override RC CurrentTimeInt64(ref long now)
        {
            // FILETIME structure is a 64-bit value representing the number of 100-nanosecond intervals since January 1, 1601 (= JD 2305813.5).
#if WINRT
            const long winRtEpoc = 17214255 * (long)8640000;
#else
            const long winFiletimeEpoch = 23058135 * (long)8640000;
#endif
#if TEST
            const long unixEpoch = 24405875 * (long)8640000;
#endif
#if WINRT
            now = winRtEpoc + DateTime.UtcNow.Ticks / (long)10000;
#else
            now = winFiletimeEpoch + DateTime.UtcNow.ToFileTimeUtc() / (long)10000;
#endif
#if TEST
#if !TCLSH
            if (current_time != 0)
                now = 1000 * (long)current_time + unixEpoch;
#else
            if (current_time.iValue != 0)
                now = 1000 * (long)current_time.iValue + unixEpoch;
#endif
#endif
            return RC.OK;
        }

        public override RC CurrentTime(ref double now)
        {
            long i = 0;
            var rc = CurrentTimeInt64(ref i);
            if (rc == RC.OK)
                now = i / 86400000.0;
            return rc;
        }

        public override RC GetLastError(int bufLength, ref string buf)
        {
            return getLastErrorMsg(ref buf);
        }

        #endregion
    }

    #region Bootstrap VSystem

    public abstract partial class VSystem
    {
        public static RC Initialize()
        {
            RegisterVfs(new WinVSystem(), true, () => new WinVSystem.WinVFile());
            return RC.OK;
        }

        public static void Shutdown()
        {
        }
    }

    #endregion
}