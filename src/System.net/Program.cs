using Core;
using Core.IO;
using System;

namespace GpuData
{
    class MainClass
    {
        public static void Main(string[] args)
        {
            MutexEx masterMutex;
            RC rc = SysEx.PreInitialize(out masterMutex);
            if (rc != RC.OK) return;
            SysEx.PostInitialize(masterMutex);
            //TestVFS();
            //Console.ReadKey();
        }

        static void TestVFS()
        {
            var vfs = VSystem.FindVfs("win32");
            if (vfs == null)
                throw new InvalidOperationException();
            VFile file;
            VSystem.OPEN flagOut;
            var rc = vfs.OpenAndAlloc(@"C:\T_\Test.db", out file, VSystem.OPEN.CREATE | VSystem.OPEN.READWRITE | VSystem.OPEN.MAIN_DB, out flagOut);
            if (rc != RC.OK)
                throw new InvalidOperationException();
            file.Write4(0, 12345);
            file.Close();
        }
    }
}
