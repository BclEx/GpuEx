namespace Core
{
    public partial class BContext
    {
        #region Busy Handler

        public int InvokeBusyHandler()
        {
            if (C._NEVER(BusyHandler == null) || BusyHandler.Func == null || BusyHandler.Busys < 0) return 0;
            int rc = BusyHandler.Func(BusyHandler.Arg, BusyHandler.Busys);
            if (rc == 0)
                BusyHandler.Busys = -1;
            else
                BusyHandler.Busys++;
            return rc;
        }

        #endregion
    }
}
