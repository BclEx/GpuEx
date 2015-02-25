using Pid = System.UInt32;

namespace Core
{
    #region IBackup

    public interface IBackup
    {
        void Update(Pid id, byte[] data);
        void Restart();
    }

    #endregion
}