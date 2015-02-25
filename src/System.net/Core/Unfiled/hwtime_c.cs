// hwtime.c
namespace Core
{
    public partial class Sqlite3
    {
        static long sqlite3Hwtime()
        {
            return (long)System.DateTime.Now.Ticks;
        }
    }
}
