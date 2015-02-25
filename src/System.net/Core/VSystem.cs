using Core.IO;
using System;
using System.Diagnostics;
using System.Text;
namespace Core
{
    public abstract partial class VSystem
    {
        internal static VSystem _vfsList;
        //internal static bool _isInit = false;

        public enum OPEN : uint
        {
            READONLY = 0x00000001,          // Ok for sqlite3_open_v2() 
            READWRITE = 0x00000002,         // Ok for sqlite3_open_v2() 
            CREATE = 0x00000004,            // Ok for sqlite3_open_v2() 
            DELETEONCLOSE = 0x00000008,     // VFS only 
            EXCLUSIVE = 0x00000010,         // VFS only 
            AUTOPROXY = 0x00000020,         // VFS only 
            URI = 0x00000040,               // Ok for sqlite3_open_v2() 
            MEMORY = 0x00000080,            // Ok for sqlite3_open_v2()
            MAIN_DB = 0x00000100,           // VFS only 
            TEMP_DB = 0x00000200,           // VFS only 
            TRANSIENT_DB = 0x00000400,      // VFS only 
            MAIN_JOURNAL = 0x00000800,      // VFS only 
            TEMP_JOURNAL = 0x00001000,      // VFS only 
            SUBJOURNAL = 0x00002000,        // VFS only 
            MASTER_JOURNAL = 0x00004000,    // VFS only 
            NOMUTEX = 0x00008000,           // Ok for sqlite3_open_v2() 
            FULLMUTEX = 0x00010000,         // Ok for sqlite3_open_v2() 
            SHAREDCACHE = 0x00020000,       // Ok for sqlite3_open_v2() 
            PRIVATECACHE = 0x00040000,      // Ok for sqlite3_open_v2() 
            WAL = 0x00080000,               // VFS only 
        }

        public enum ACCESS : byte
        {
            EXISTS = 0,
            READWRITE = 1,  // Used by PRAGMA temp_store_directory
            READ = 2,       // Unused
        }

        public VSystem Next;        // Next registered VFS
        public string Name = "win32";   // Name of this virtual file system
        public object Tag;              // Pointer to application-specific data
        public int SizeOsFile = -1;     // Size of subclassed VirtualFile
        public int MaxPathname = 256;   // Maximum file pathname length
        public Func<VFile> CreateOsFile;

        public void _memcpy(VSystem ct)
        {
            ct.SizeOsFile = this.SizeOsFile;
            ct.MaxPathname = this.MaxPathname;
            ct.Next = this.Next;
            ct.Name = this.Name;
            ct.Tag = this.Tag;
        }

        public abstract RC Open(string path, VFile file, OPEN flags, out OPEN outFlags);
        public abstract RC Delete(string path, bool syncDirectory);
        public abstract RC Access(string path, ACCESS flags, out int outRC);
        public abstract RC FullPathname(string path, out string outPath);

        public abstract object DlOpen(string filename);
        public abstract void DlError(int bufLength, string buf);
        public abstract object DlSym(object handle, string symbol);
        public abstract void DlClose(object handle);

        public abstract int Randomness(int bufLength, byte[] buf);
        public abstract int Sleep(int microseconds);
        public abstract RC CurrentTimeInt64(ref long now);
        public abstract RC CurrentTime(ref double now);
        public abstract RC GetLastError(int bufLength, ref string buf);

        public RC OpenAndAlloc(string path, out VFile file, OPEN flags, out OPEN outFlags)
        {
            file = null;
            outFlags = 0;
            VFile file2 = CreateOsFile();
            if (file2 == null)
                return RC.NOMEM;
            RC rc = Open(path, file2, flags, out outFlags);
            if (rc != RC.OK)
                C._free(ref file2);
            else
                file = file2;
            return rc;
        }

        public static VSystem FindVfs(string name)
        {
            VSystem vfs = null;
            var mutex = MutexEx.Alloc(MutexEx.MUTEX.STATIC_MASTER);
            MutexEx.Enter(mutex);
            for (vfs = _vfsList; vfs != null && name != vfs.Name; vfs = vfs.Next) { }
            MutexEx.Leave(mutex);
            return vfs;
        }

        internal static void UnlinkVfs(VSystem vfs)
        {
            Debug.Assert(MutexEx.Held(MutexEx.Alloc(MutexEx.MUTEX.STATIC_MASTER)));
            if (vfs == null) { }
            else if (_vfsList == vfs)
                _vfsList = vfs.Next;
            else if (_vfsList != null)
            {
                var p = _vfsList;
                while (p.Next != null && p.Next != vfs)
                    p = p.Next;
                if (p.Next == vfs)
                    p.Next = vfs.Next;
            }
        }

        public static RC RegisterVfs(VSystem vfs, bool default_, Func<VFile> createOsFile)
        {
            var mutex = MutexEx.Alloc(MutexEx.MUTEX.STATIC_MASTER);
            MutexEx.Enter(mutex);
            UnlinkVfs(vfs);
            vfs.CreateOsFile = createOsFile;
            if (default_ || _vfsList == null)
            {
                vfs.Next = _vfsList;
                _vfsList = vfs;
            }
            else
            {
                vfs.Next = _vfsList.Next;
                _vfsList.Next = vfs;
            }
            Debug.Assert(_vfsList != null);
            MutexEx.Leave(mutex);
            return RC.OK;
        }

        public static RC UnregisterVfs(VSystem vfs)
        {
            var mutex = MutexEx.Alloc(MutexEx.MUTEX.STATIC_MASTER);
            MutexEx.Enter(mutex);
            UnlinkVfs(vfs);
            MutexEx.Leave(mutex);
            return RC.OK;
        }

        #region File

#if ENABLE_8_3_NAMES
        public static void FileSuffix3(string baseFilename, ref string z)
        {
#if ENABLE_8_3_NAMESx2
		if (!UriBoolean(baseFilename, "8_3_names", 0)) return;
#endif
            int sz = z.Length;
            int i;
            for (i = sz - 1; i > 0 && z[i] != '/' && z[i] != '.'; i--) { }
            //if (z[i] == '.' && C._ALWAYS(sz > i + 4)) C._memmove(&z[i + 1], &z[sz - 3], 4);
            //if (z[i] == '.' && C._ALWAYS(sz > i + 4)) C._memcpy(&z[i + 1], &z[sz - 3], 4);
        }
#else
        public static void FileSuffix3(string baseFilename, ref string z) { }
#endif

        class OpenMode
        {
            public string Z;
            public VSystem.OPEN Mode;

            public OpenMode(string z, VSystem.OPEN mode)
            {
                Z = z;
                Mode = mode;
            }
        }

        static readonly OpenMode[] _cacheModes = new OpenMode[]
        {
           new OpenMode("shared",  VSystem.OPEN.SHAREDCACHE),
           new OpenMode("private", VSystem.OPEN.PRIVATECACHE),
           new OpenMode(null, (VSystem.OPEN)0)
        };

        static readonly OpenMode[] _openModes = new OpenMode[]
        {
            new OpenMode("ro",  VSystem.OPEN.READONLY),
            new OpenMode("rw",  VSystem.OPEN.READWRITE), 
            new OpenMode("rwc", VSystem.OPEN.READWRITE | VSystem.OPEN.CREATE),
            new OpenMode(null, (VSystem.OPEN)0)
        };

        public static RC ParseUri(string defaultVfsName, string uri, ref VSystem.OPEN flagsRef, out VSystem vfsOut, out string fileNameOut, out string errMsgOut)
        {
            vfsOut = null;
            fileNameOut = null;
            errMsgOut = null;

            VSystem.OPEN flags = flagsRef;
            string vfsName = defaultVfsName;
            int uriLength = uri.Length;

            RC rc = RC.OK;
            StringBuilder fileName = null;

            if (((flags & VSystem.OPEN.URI) != 0 || SysEx._GlobalStatics.OpenUri) && uriLength >= 5 && uri.StartsWith("file:"))
            {
                // Make sure the SQLITE_OPEN_URI flag is set to indicate to the VFS xOpen method that there may be extra parameters following the file-name.
                flags |= VSystem.OPEN.URI;

                int bytes = uriLength + 2; // Bytes of space to allocate               
                int uriIdx; // Input character index
                for (uriIdx = 0; uriIdx < uriLength; uriIdx++) bytes += (uri[uriIdx] == '&' ? 1 : 0);
                fileName = new StringBuilder(bytes);
                if (fileName == null) return RC.NOMEM;

                // Discard the scheme and authority segments of the URI.
                if (uri[5] == '/' && uri[6] == '/')
                {
                    uriIdx = 7;
                    while (uriIdx < uriLength && uri[uriIdx] != '/') uriIdx++;
                    if (uriIdx != 7 && (uriIdx != 16 || !string.Equals("localhost", uri.Substring(7, 9), StringComparison.InvariantCultureIgnoreCase)))
                    {
                        errMsgOut = C._mprintf("invalid uri authority: %.*s", uriIdx - 7, uri.Substring(7));
                        rc = RC.ERROR;
                        goto parse_uri_out;
                    }
                }
                else
                    uriIdx = 5;

                // Copy the filename and any query parameters into the zFile buffer. Decode %HH escape codes along the way. 
                //
                // Within this loop, variable eState may be set to 0, 1 or 2, depending on the parsing context. As follows:
                //
                //   0: Parsing file-name.
                //   1: Parsing name section of a name=value query parameter.
                //   2: Parsing value section of a name=value query parameter.
                int state = 0; // Parser state when parsing URI
                char c;
                //int fileNameIdx = 0; // Output character index
                while (uriIdx < uriLength && (c = uri[uriIdx]) != 0 && c != '#')
                {
                    uriIdx++;

                    if (c == '%' && C._isxdigit(uri[uriIdx]) && C._isxdigit(uri[uriIdx + 1]))
                    {
                        int octet = (C._hextobyte(uri[uriIdx++]) << 4);
                        octet += C._hextobyte(uri[uriIdx++]);
                        Debug.Assert(octet >= 0 && octet < 256);
                        if (octet == 0)
                        {
                            // This branch is taken when "%00" appears within the URI. In this case we ignore all text in the remainder of the path, name or
                            // value currently being parsed. So ignore the current character and skip to the next "?", "=" or "&", as appropriate.
                            while (uriIdx < uriLength && (c = uri[uriIdx]) != 0 && c != '#' &&
                                (state != 0 || c != '?') &&
                                (state != 1 || (c != '=' && c != '&')) &&
                                (state != 2 || c != '&'))
                                uriIdx++;
                            continue;
                        }
                        c = (char)octet;
                    }
                    else if (state == 1 && (c == '&' || c == '='))
                    {
                        if (fileName[fileName.Length - 1] == '\0')
                        {
                            // An empty option name. Ignore this option altogether.
                            while (uri[uriIdx] != '\0' && uri[uriIdx] != '#' && uri[uriIdx - 1] != '&') uriIdx++;
                            continue;
                        }
                        if (c == '&')
                            fileName.Append('\0');
                        else
                            state = 2;
                        c = '\0';
                    }
                    else if ((state == 0 && c == '?') || (state == 2 && c == '&'))
                    {
                        c = '\0';
                        state = 1;
                    }
                    fileName.Append(c);
                }
                if (state == 1) fileName.Append('\0');
                fileName.Append('\0');
                fileName.Append('\0');

                // Check if there were any options specified that should be interpreted here. Options that are interpreted here include "vfs" and those that
                // correspond to flags that may be passed to the sqlite3_open_v2() method.
                string opt = fileName.ToString().Substring(fileName.Length + 1);
                while (opt.Length > 0)
                {
                    int optLength = opt.Length;
                    string val = opt.Substring(optLength);
                    int valLength = val.Length;
                    if (optLength == 3 && opt.StartsWith("vfs"))
                        vfsName = val;
                    else
                    {
                        OpenMode[] modes = null;
                        string modeType = null;
                        VSystem.OPEN mask = (VSystem.OPEN)0;
                        VSystem.OPEN limit = (VSystem.OPEN)0;
                        if (optLength == 5 && opt.StartsWith("cache"))
                        {
                            mask = VSystem.OPEN.SHAREDCACHE | VSystem.OPEN.PRIVATECACHE;
                            modes = _cacheModes;
                            limit = mask;
                            modeType = "cache";
                        }
                        if (optLength == 4 && opt.StartsWith("mode"))
                        {
                            mask = VSystem.OPEN.READONLY | VSystem.OPEN.READWRITE | VSystem.OPEN.CREATE;
                            modes = _openModes;
                            limit = mask & flags;
                            modeType = "access";
                        }
                        if (modes != null)
                        {
                            VSystem.OPEN mode = 0;
                            for (int i = 0; modes[i].Z != null; i++)
                            {
                                string z = modes[i].Z;
                                if (valLength == z.Length && z.StartsWith(val))
                                {
                                    mode = modes[i].Mode;
                                    break;
                                }
                            }
                            if (mode == 0)
                            {
                                errMsgOut = C._mprintf("no such %s mode: %s", modeType, val);
                                rc = RC.ERROR;
                                goto parse_uri_out;
                            }
                            if (mode > limit)
                            {
                                errMsgOut = C._mprintf("%s mode not allowed: %s", modeType, val);
                                rc = RC.PERM;
                                goto parse_uri_out;
                            }
                            flags = ((flags & ~mask) | mode);
                        }
                    }
                    opt = val.Substring(valLength + 1);
                }
            }
            else
            {
                fileName = (uri == null ? new StringBuilder() : new StringBuilder(uri.Substring(0, uriLength)));
                if (fileName == null) return RC.NOMEM;
                fileName.Append('\0');
                fileName.Append('\0');
            }

            vfsOut = FindVfs(vfsName);
            if (vfsOut == null)
            {
                errMsgOut = C._mprintf("no such vfs: %s", vfsName);
                rc = RC.ERROR;
            }
        parse_uri_out:
            if (rc != RC.OK)
            {
                C._free(ref fileName);
                fileName = null;
            }
            flagsRef = flags;
            fileNameOut = (fileName == null ? null : fileName.ToString().Substring(0, fileName.Length));
            return rc;
        }

        public static string UriParameter(string filename, string param)
        {
            if (filename == null) return null;
            //int filenameIdx _strlen30(filename) + 1;
            //while (filename[0])
            //{
            //    int x = _strcmp(filename, param);
            //    filename += _strlen30(filename) + 1;
            //    if (x == 0) return filename;
            //    filename += _strlen30(filename) + 1;
            //}
            return null;
        }

        public static bool UriBoolean(string filename, string param, bool dflt)
        {
            string z = UriParameter(filename, param);
            return (z != null ? ConvertEx.GetBoolean(z, (byte)(dflt ? 1 : 0)) : dflt);
        }

        public static long UriInt64(string filename, string param, long dflt)
        {
            string z = UriParameter(filename, param);
            long v;
            return (z != null && ConvertEx.Atoi64(z, out v, z.Length, TEXTENCODE.UTF8) == 0 ? v : dflt);
        }

        #endregion
    }
}