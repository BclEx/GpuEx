using Core.IO;
using Pid = System.UInt32;

namespace Core
{
    public partial class Wal
    {
        internal static RC Open(VSystem x, VFile y, string z) { return RC.OK; }
        internal void Limit(long y) { }
        internal RC Close(int x, int y, byte z) { return 0; }
        internal RC BeginReadTransaction(int z) { return 0; }
        internal void EndReadTransaction() { }
        internal RC Read(Pid w, ref int x, int y, byte[] z) { return 0; }
        internal Pid DBSize() { return 0; }
        internal RC BeginWriteTransaction() { return 0; }
        internal RC EndWriteTransaction() { return 0; }
        internal RC Undo(int y, object z) { return 0; }
        internal void Savepoint(object z) { }
        internal RC SavepointUndo(object z) { return 0; }
        internal RC Frames(int v, PgHdr w, Pid x, int y, int z) { return 0; }
        internal RC Checkpoint(int s, int t, byte[] u, int v, int w, byte[] x, ref int y, ref int z) { y = 0; z = 0; return 0; }
        internal int get_Callback() { return 0; }
        internal bool ExclusiveMode(int z) { return false; }
        internal bool get_HeapMemory() { return false; }
#if ENABLE_ZIPVFS
        internal int get_Framesize() { return 0; }
#endif
    }
}
