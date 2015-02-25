using System;
using System.Diagnostics;
using System.Text;
using time_t = System.Int64;

namespace Core.Command
{
    public partial class Date_
    {
        #region OMIT_DATETIME_FUNCS
#if !OMIT_DATETIME_FUNCS

        // Temporary String for use in this module
        static StringBuilder _zdtTemp = new StringBuilder(100);
        static StringBuilder _zdtBuf = new StringBuilder(100);

        public class tm
        {
            public int tm_sec;     // seconds after the minute - [0,59]
            public int tm_min;     // minutes after the hour - [0,59]
            public int tm_hour;    // hours since midnight - [0,23]
            public int tm_mday;    // day of the month - [1,31]
            public int tm_mon;     // months since January - [0,11]
            public int tm_year;    // years since 1900
            public int tm_wday;    // days since Sunday - [0,6]
            public int tm_yday;    // days since January 1 - [0,365]
            public int tm_isdst;   // daylight savings time flag
        }

        public class DateTime
        {
            public long JD; // The julian day number times 86400000
            public int Y, M, D;       // Year, month, and day
            public int h, m;          // Hour and minutes
            public int tz;            // Timezone offset in minutes
            public double s;          // Seconds
            public bool ValidYMD;     // True (1) if Y,M,D are valid
            public bool ValidHMS;     // True (1) if h,m,s are valid
            public bool ValidJD;      // True (1) if iJD is valid
            public bool ValidTZ;      // True (1) if tz is valid

            public void memcopy(DateTime ct)
            {
                ct.JD = JD;
                ct.Y = Y;
                ct.M = M;
                ct.D = D;
                ct.h = h;
                ct.m = m;
                ct.tz = tz;
                ct.s = s;
                ct.ValidYMD = ValidYMD;
                ct.ValidHMS = ValidHMS;
                ct.ValidJD = ValidJD;
                ct.ValidTZ = ValidJD;
            }
        }

        static string GetDigitsStep(string date, int n, int min, int max, char nextC, out int value)
        {
            int val = 0;
            int dateIdx = 0;
            string nextDate = null;
            while (n-- != 0)
            {
                if (!char.IsDigit(date[dateIdx])) goto end_getDigits;
                val = val * 10 + date[dateIdx] - '0';
                dateIdx++;
            }
            if (val < min || val > max || dateIdx < date.Length && (nextC != 0 && nextC != date[dateIdx])) goto end_getDigits;
            nextDate = date.Substring(++dateIdx);

        end_getDigits:
            value = val;
            return nextDate;
        }
        static int GetDigits(string date,
            int n0, int min0, int max0, char nextC0, out int value0)
        {
            date = GetDigitsStep(date, n0, min0, max0, nextC0, out value0);
            return (date == null ? 0 : 1);
        }
        static int GetDigits(string date,
            int n0, int min0, int max0, char nextC0, out int value0,
            int n1, int min1, int max1, char nextC1, out int value1)
        {
            date = GetDigitsStep(date, n0, min0, max0, nextC0, out value0);
            if (date == null) { value1 = 0; return 0; }
            date = GetDigitsStep(date, n1, min1, max1, nextC1, out value1);
            return (date == null ? 1 : 2);
        }
        static int GetDigits(string date,
            int n0, int min0, int max0, char nextC0, out int value0,
            int n1, int min1, int max1, char nextC1, out int value1,
            int n2, int min2, int max2, char nextC2, out int value2)
        {
            date = GetDigitsStep(date, n0, min0, max0, nextC0, out value0);
            if (date == null) { value1 = 0; value2 = 0; return 0; }
            date = GetDigitsStep(date, n1, min1, max1, nextC1, out value1);
            if (date == null) { value2 = 0; return 1; }
            date = GetDigitsStep(date, n2, min2, max2, nextC2, out value2);
            return (date == null ? 2 : 3);
        }

        static bool ParseTimezone(string date, DateTime p)
        {
            date = date.Trim();
            p.tz = 0;
            int sgn = 0;
            int c = (date.Length == 0 ? '\0' : date[0]);
            if (c == '-') sgn = -1;
            else if (c == '+') sgn = +1;
            else if (c == 'Z' || c == 'z')
            {
                date = date.Substring(1);
                goto zulu_time;
            }
            else return (c != '\0');
            date = date.Substring(1);
            int hrs, mns;
            if (GetDigits(date, 2, 0, 14, ':', out hrs, 2, 0, 59, '\0', out mns) != 2)
                return true;
            date = (date.Length <= 6 ? string.Empty : date.Substring(6));
            p.tz = sgn * (mns + hrs * 60);
        zulu_time:
            date = date.Trim();
            return (date != string.Empty);
        }

        static bool ParseHhMmSs(string date, DateTime p)
        {
            int h, m, s;
            double ms = 0.0;
            if (GetDigits(date, 2, 0, 24, ':', out h, 2, 0, 59, '\0', out m) != 2)
                return true;
            int dateIdx = 5;
            if (dateIdx < date.Length && date[dateIdx] == ':')
            {
                dateIdx++;
                if (GetDigits(date.Substring(dateIdx), 2, 0, 59, '\0', out s) != 1)
                    return true;
                dateIdx += 2;
                if (dateIdx + 1 < date.Length && date[dateIdx] == '.' && char.IsDigit(date[dateIdx + 1]))
                {
                    double scale = 1.0;
                    dateIdx++;
                    while (dateIdx < date.Length && char.IsDigit(date[dateIdx]))
                    {
                        ms = ms * 10.0 + date[dateIdx] - '0';
                        scale *= 10.0;
                        dateIdx++;
                    }
                    ms /= scale;
                }
            }
            else
                s = 0;
            p.ValidJD = false;
            p.ValidHMS = true;
            p.h = h;
            p.m = m;
            p.s = s + ms;
            if (dateIdx < date.Length && ParseTimezone(date.Substring(dateIdx), p)) return true;
            p.ValidTZ = (p.tz != 0);
            return false;
        }

        static void ComputeJD(DateTime p)
        {
            if (p.ValidJD) return;
            int Y, M, D;
            if (p.ValidYMD)
            {
                Y = p.Y;
                M = p.M;
                D = p.D;
            }
            else
            {
                Y = 2000; // If no YMD specified, assume 2000-Jan-01
                M = 1;
                D = 1;
            }
            if (M <= 2)
            {
                Y--;
                M += 12;
            }
            int A = Y / 100;
            int B = 2 - A + (A / 4);
            int X1 = (int)(36525 * (Y + 4716) / 100);
            int X2 = (int)(306001 * (M + 1) / 10000);
            p.JD = (long)((X1 + X2 + D + B - 1524.5) * 86400000);
            p.ValidJD = true;
            if (p.ValidHMS)
            {
                p.JD += (long)(p.h * 3600000 + p.m * 60000 + p.s * 1000);
                if (p.ValidTZ)
                {
                    p.JD -= p.tz * 60000;
                    p.ValidYMD = false;
                    p.ValidHMS = false;
                    p.ValidTZ = false;
                }
            }
        }

        static bool ParseYyyyMmDd(string date, DateTime p)
        {
            int dateIdx = 0;
            bool neg;
            if (date[dateIdx] == '-')
            {
                dateIdx++;
                neg = true;
            }
            else
                neg = false;
            int Y, M, D;
            if (GetDigits(date.Substring(dateIdx), 4, 0, 9999, '-', out Y, 2, 1, 12, '-', out M, 2, 1, 31, '\0', out D) != 3)
                return true;
            dateIdx += 10;
            while (dateIdx < date.Length && (char.IsWhiteSpace(date[dateIdx]) || 'T' == date[dateIdx])) { dateIdx++; }
            if (dateIdx < date.Length && !ParseHhMmSs(date.Substring(dateIdx), p)) { } // We got the time
            else if (dateIdx >= date.Length) p.ValidHMS = false;
            else return true;
            p.ValidJD = false;
            p.ValidYMD = true;
            p.Y = (neg ? -Y : Y);
            p.M = M;
            p.D = D;
            if (p.ValidTZ)
                ComputeJD(p);
            return false;
        }

        static bool SetDateTimeToCurrent(FuncContext fctx, DateTime p)
        {
            Context ctx = Vdbe.Context_Ctx(fctx);
            if (ctx.Vfs.CurrentTimeInt64(ref p.JD) == RC.OK)
            {
                p.ValidJD = true;
                return false;
            }
            return true;
        }

        static bool ParseDateOrTime(FuncContext fctx, string date, ref DateTime p)
        {
            double r = 0.0;
            if (!ParseYyyyMmDd(date, p)) return false;
            else if (!ParseHhMmSs(date, p)) return false;
            else if (date.Equals("now", StringComparison.InvariantCultureIgnoreCase)) return SetDateTimeToCurrent(fctx, p);
            else if (ConvertEx.Atof(date, ref r, date.Length, TEXTENCODE.UTF8))
            {
                p.JD = (long)(r * 86400000.0 + 0.5);
                p.ValidJD = true;
                return false;
            }
            return true;
        }

        static void ComputeYMD(DateTime p)
        {
            if (p.ValidYMD) return;
            if (!p.ValidJD)
            {
                p.Y = 2000;
                p.M = 1;
                p.D = 1;
            }
            else
            {
                int Z = (int)((p.JD + 43200000) / 86400000);
                int A = (int)((Z - 1867216.25) / 36524.25);
                A = Z + 1 + A - (A / 4);
                int B = A + 1524;
                int C = (int)((B - 122.1) / 365.25);
                int D = (int)((36525 * C) / 100);
                int E = (int)((B - D) / 30.6001);
                int X1 = (int)(30.6001 * E);
                p.D = B - D - X1;
                p.M = (E < 14 ? E - 1 : E - 13);
                p.Y = (p.M > 2 ? C - 4716 : C - 4715);
            }
            p.ValidYMD = true;
        }

        static void ComputeHMS(DateTime p)
        {
            if (p.ValidHMS) return;
            ComputeJD(p);
            int s = (int)((p.JD + 43200000) % 86400000);
            p.s = s / 1000.0;
            s = (int)p.s;
            p.s -= s;
            p.h = s / 3600;
            s -= p.h * 3600;
            p.m = s / 60;
            p.s += s - p.m * 60;
            p.ValidHMS = true;
        }

        static void ComputeYMD_HMS(DateTime p)
        {
            ComputeYMD(p);
            ComputeHMS(p);
        }

        static void ClearYMD_HMS_TZ(DateTime p)
        {
            p.ValidYMD = false;
            p.ValidHMS = false;
            p.ValidTZ = false;
        }


#if !OMIT_LOCALTIME
        static int OsLocaltime(time_t t, tm tm_)
        {
            int rc;
            MutexEx mutex = MutexEx.Alloc(MutexEx.MUTEX.STATIC_MASTER);
            MutexEx.Enter(mutex);
            tm x = localtime(t);
#if !OMIT_BUILTIN_TEST
            if (_localtimeFault) x = null;
#endif
            if (x != null) tm_ = x;
            MutexEx.Leave(mutex);
            rc = (x == null ? 1 : 0);
            return rc;
        }

        static long LocaltimeOffset(DateTime p, FuncContext fctx, out RC rc)
        {
            // Initialize the contents of sLocal to avoid a compiler warning.
            tm sLocal = new tm();

            DateTime x = p;
            ComputeYMD_HMS(x);
            if (x.Y < 1971 || x.Y >= 2038)
            {
                x.Y = 2000;
                x.M = 1;
                x.D = 1;
                x.h = 0;
                x.m = 0;
                x.s = 0.0;
            }
            else
            {
                int s = (int)(x.s + 0.5);
                x.s = s;
            }
            x.tz = 0;
            x.ValidJD = false;
            ComputeJD(x);
            time_t t = (long)(x.JD / 1000 - 210866760000L);
            if (OsLocaltime(t, sLocal) != 0)
            {
                Vdbe.Result_Error(fctx, "local time unavailable", -1);
                rc = RC.ERROR;
                return 0;
            }
            DateTime y = new DateTime();
            y.Y = sLocal.tm_year; // +1900;
            y.M = sLocal.tm_mon; // +1;
            y.D = sLocal.tm_mday;
            y.h = sLocal.tm_hour;
            y.m = sLocal.tm_min;
            y.s = sLocal.tm_sec;
            y.ValidYMD = true;
            y.ValidHMS = true;
            y.ValidJD = false;
            y.ValidTZ = false;
            ComputeJD(y);
            rc = RC.OK;
            return (int)(y.JD - x.JD);
        }
#endif

        static RC ParseModifier(FuncContext fctx, string mod, DateTime p)
        {
            RC rc = RC.ERROR;
            int n;
            double r = 0;
            StringBuilder z = new StringBuilder(mod.ToLower());
            _zdtBuf.Length = 0;
            switch (z[0])
#if !OMIT_LOCALTIME
            {
                case 'l':
                    {
                        // localtime - Assuming the current time value is UTC (a.k.a. GMT), shift it to show local time.
                        if (z.ToString() == "localtime")
                        {
                            ComputeJD(p);
                            p.JD += LocaltimeOffset(p, fctx, out rc);
                            ClearYMD_HMS_TZ(p);
                        }
                        break;
                    }
#endif
                case 'u':
                    {
                        // unixepoch - Treat the current value of p->iJD as the number of seconds since 1970.  Convert to a real julian day number.
                        if (z.ToString() == "unixepoch" && p.ValidJD)
                        {
                            p.JD = (long)((p.JD + 43200) / 86400 + 210866760000000L);
                            ClearYMD_HMS_TZ(p);
                            rc = RC.OK;
                        }
#if !OMIT_LOCALTIME
                        else if (z.ToString() == "utc")
                        {
                            ComputeJD(p);
                            long c1 = LocaltimeOffset(p, fctx, out rc);
                            if (rc == RC.OK)
                            {
                                p.JD -= c1;
                                ClearYMD_HMS_TZ(p);
                                p.JD += c1 - LocaltimeOffset(p, fctx, out rc);
                            }
                        }
#endif
                        break;
                    }
                case 'w':
                    {
                        // weekday N - Move the date to the same time on the next occurrence of weekday N where 0==Sunday, 1==Monday, and so forth.  If the date is already on the appropriate weekday, this is a no-op.
                        if (z.ToString().StartsWith("weekday ") && ConvertEx.Atof(z.ToString().Substring(8), ref r, z.ToString().Substring(8).Length, TEXTENCODE.UTF8) && (n = (int)r) == r && n >= 0 && r < 7)
                        {
                            ComputeYMD_HMS(p);
                            p.ValidTZ = false;
                            p.ValidJD = false;
                            ComputeJD(p);
                            long Z = ((p.JD + 129600000) / 86400000) % 7;
                            if (Z > n) Z -= 7;
                            p.JD += (n - Z) * 86400000;
                            ClearYMD_HMS_TZ(p);
                            rc = RC.OK;
                        }
                        break;
                    }
                case 's':
                    {
                        // start of TTTTT - Move the date backwards to the beginning of the current day, or month or year.
                        if (z.Length <= 9) z.Length = 0;
                        else z.Remove(0, 9);
                        ComputeYMD(p);
                        p.ValidHMS = true;
                        p.h = p.m = 0;
                        p.s = 0.0;
                        p.ValidTZ = false;
                        p.ValidJD = false;
                        if (z.ToString() == "month")
                        {
                            p.D = 1;
                            rc = RC.OK;
                        }
                        else if (z.ToString() == "year")
                        {
                            ComputeYMD(p);
                            p.M = 1;
                            p.D = 1;
                            rc = RC.OK;
                        }
                        else if (z.ToString() == "day")
                            rc = RC.OK;
                        break;
                    }
                case '+':
                case '-':
                case '0':
                case '1':
                case '2':
                case '3':
                case '4':
                case '5':
                case '6':
                case '7':
                case '8':
                case '9':
                    {
                        for (n = 1; n < z.Length && z[n] != ':' && !char.IsWhiteSpace(z[n]); n++) { }
                        if (!ConvertEx.Atof(z.ToString(), ref r, n, TEXTENCODE.UTF8))
                        {
                            rc = RC.ERROR;
                            break;
                        }
                        if (z[n] == ':')
                        {
                            // A modifier of the form (+|-)HH:MM:SS.FFF adds (or subtracts) the specified number of hours, minutes, seconds, and fractional seconds
                            // to the time.  The ".FFF" may be omitted.  The ":SS.FFF" may be omitted.
                            string z2 = z.ToString();
                            int z2Idx = 0;
                            if (!char.IsWhiteSpace(z2[z2Idx])) z2Idx++;
                            DateTime tx = new DateTime();
                            if (ParseHhMmSs(z2.Substring(z2Idx), tx)) break;
                            ComputeJD(tx);
                            tx.JD -= 43200000;
                            long day = tx.JD / 86400000;
                            tx.JD -= day * 86400000;
                            if (z[0] == '-') tx.JD = -tx.JD;
                            ComputeJD(p);
                            ClearYMD_HMS_TZ(p);
                            p.JD += tx.JD;
                            rc = RC.OK;
                            break;
                        }
                        while (char.IsWhiteSpace(z[n])) n++; z = z.Remove(0, n);
                        n = z.Length;
                        if (n > 10 || n < 3) break;
                        if (z[n - 1] == 's') z.Length = --n;
                        ComputeJD(p);
                        rc = RC.OK;
                        double rounder = (r < 0 ? -0.5 : +0.5);
                        if (n == 3 && z.ToString() == "day") p.JD += (long)(r * 86400000.0 + rounder);
                        else if (n == 4 && z.ToString() == "hour") p.JD += (long)(r * (86400000.0 / 24.0) + rounder);
                        else if (n == 6 && z.ToString() == "minute") p.JD += (long)(r * (86400000.0 / (24.0 * 60.0)) + rounder);
                        else if (n == 6 && z.ToString() == "second") p.JD += (long)(r * (86400000.0 / (24.0 * 60.0 * 60.0)) + rounder);
                        else if (n == 5 && z.ToString() == "month")
                        {
                            ComputeYMD_HMS(p);
                            p.M += (int)r;
                            int x = (p.M > 0 ? (p.M - 1) / 12 : (p.M - 12) / 12);
                            p.Y += x;
                            p.M -= x * 12;
                            p.ValidJD = false;
                            ComputeJD(p);
                            int y = (int)r;
                            if (y != r)
                                p.JD += (long)((r - y) * 30.0 * 86400000.0 + rounder);
                        }
                        else if (n == 4 && z.ToString() == "year")
                        {
                            int y = (int)r;
                            ComputeYMD_HMS(p);
                            p.Y += y;
                            p.ValidJD = false;
                            ComputeJD(p);
                            if (y != r)
                                p.JD += (long)((r - y) * 365.0 * 86400000.0 + rounder);
                        }
                        else
                            rc = RC.ERROR;
                        ClearYMD_HMS_TZ(p);
                        break;
                    }
                default: break;
            }
            return rc;
        }

        static bool IsDate(FuncContext fctx, int argc, Mem[] argv, out DateTime p)
        {
            int i;
            string z;
            p = new DateTime();
            if (argc == 0)
                SetDateTimeToCurrent(fctx, p);
            TYPE type;
            if ((type = Vdbe.Value_Type(argv[0])) == TYPE.FLOAT || type == TYPE.INTEGER)
            {
                p.JD = (long)(Vdbe.Value_Double(argv[0]) * 86400000.0 + 0.5);
                p.ValidJD = true;
            }
            else
            {
                z = Vdbe.Value_Text(argv[0]);
                if (z == null || ParseDateOrTime(fctx, z, ref p)) return true;
            }
            for (i = 1; i < argc; i++)
            {
                z = Vdbe.Value_Text(argv[i]);
                if (z == null || ParseModifier(fctx, z, p) != RC.OK) return true;
            }
            return false;
        }

        static void JuliandayFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            DateTime x;
            if (!IsDate(fctx, argc, argv, out x))
            {
                ComputeJD(x);
                Vdbe.Result_Double(fctx, x.JD / 86400000.0);
            }
        }

        static void DatetimeFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            DateTime x;
            if (!IsDate(fctx, argc, argv, out x))
            {
                _zdtBuf.Length = 0;
                ComputeYMD_HMS(x);
                C.__snprintf(_zdtBuf, 100, "%04d-%02d-%02d %02d:%02d:%02d", x.Y, x.M, x.D, x.h, x.m, (int)(x.s));
                Vdbe.Result_Text(fctx, _zdtBuf, -1, DESTRUCTOR_TRANSIENT);
            }
        }

        static void TimeFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            DateTime x;
            if (!IsDate(fctx, argc, argv, out x))
            {
                _zdtBuf.Length = 0;
                ComputeHMS(x);
                C.__snprintf(_zdtBuf, 100, "%02d:%02d:%02d", x.h, x.m, (int)x.s);
                Vdbe.Result_Text(fctx, _zdtBuf, -1, DESTRUCTOR_TRANSIENT);
            }
        }

        static void DateFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            DateTime x;
            if (!IsDate(fctx, argc, argv, out x))
            {
                _zdtBuf.Length = 0;
                ComputeYMD(x);
                C.__snprintf(_zdtBuf, 100, "%04d-%02d-%02d", x.Y, x.M, x.D);
                Vdbe.Result_Text(fctx, _zdtBuf, -1, DESTRUCTOR_TRANSIENT);
            }
        }

        static void StrftimeFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            {
                DateTime x = new DateTime();
                ulong n;
                int i, j;
                StringBuilder z;
                string fmt = Vdbe.Value_Text(argv[0]);
                StringBuilder zdtBuf = new StringBuilder(100);
                Mem[] argv1 = new Mem[argc - 1];
                for (i = 0; i < argc - 1; i++)
                    argv[i + 1]._memcpy(ref argv1[i]);
                if (fmt == null || IsDate(fctx, argc - 1, argv1, out x)) return;
                Context ctx = Vdbe.Context_Ctx(fctx);
                for (i = 0, n = 1; i < fmt.Length; i++, n++)
                {
                    if (fmt[i] == '%')
                    {
                        switch ((char)fmt[i + 1])
                        {
                            case 'd':
                            case 'H':
                            case 'm':
                            case 'M':
                            case 'S':
                            case 'W':
                                n++;
                                break;
                            // fall thru
                            case 'w':
                            case '%':
                                break;
                            case 'f':
                                n += 8;
                                break;
                            case 'j':
                                n += 3;
                                break;
                            case 'Y':
                                n += 8;
                                break;
                            case 's':
                            case 'J':
                                n += 50;
                                break;
                            default:
                                return; // ERROR.  return a NULL
                        }
                        i++;
                    }
                }
                C.ASSERTCOVERAGE(n == (ulong)(zdtBuf.Length - 1));
                C.ASSERTCOVERAGE(n == (ulong)zdtBuf.Length);
                C.ASSERTCOVERAGE(n == (ulong)ctx.Limits[(int)LIMIT.LENGTH] + 1);
                C.ASSERTCOVERAGE(n == (ulong)ctx.Limits[(int)LIMIT.LENGTH]);
                if (n < (ulong)zdtBuf.Capacity)
                    z = zdtBuf;
                else if (n > (ulong)ctx.Limits[(int)LIMIT.LENGTH])
                {
                    Vdbe.Result_ErrorOverflow(fctx);
                    return;
                }
                else
                {
                    z = new StringBuilder((int)n);
                    if (z == null)
                    {
                      Vdbe.Result_ErrorNoMem(fctx);
                      return;
                    }
                }
                ComputeJD(x);
                ComputeYMD_HMS(x);
                for (i = j = 0; i < fmt.Length; i++)
                {
                    if (fmt[i] != '%')
                        z.Append((char)fmt[i]);
                    else
                    {
                        i++;
                        _zdtTemp.Length = 0;
                        switch ((char)fmt[i])
                        {
                            case 'd': C.__snprintf(_zdtTemp, 3, "%02d", x.D); z.Append(_zdtTemp); j += 2; break;
                            case 'f':
                                {
                                    double s = x.s;
                                    if (s > 59.999) s = 59.999;
                                    C.__snprintf(_zdtTemp, 7, "%06.3f", s);
                                    z.Append(_zdtTemp);
                                    j = z.Length;
                                    break;
                                }
                            case 'H': C.__snprintf(_zdtTemp, 3, "%02d", x.h); z.Append(_zdtTemp); j += 2; break;
                            case 'W': // Fall thru
                            case 'j':
                                {
                                    DateTime y = new DateTime();
                                    x.memcopy(y);
                                    y.ValidJD = false;
                                    y.M = 1;
                                    y.D = 1;
                                    ComputeJD(y);
                                    int days = (int)((x.JD - y.JD + 43200000) / 86400000); ; // Number of days since 1st day of year
                                    if (fmt[i] == 'W')
                                    {
                                        int wd = (int)(((x.JD + 43200000) / 86400000) % 7);  // 0=Monday, 1=Tuesday, ... 6=Sunday
                                        C.__snprintf(_zdtTemp, 3, "%02d", (days + 7 - wd) / 7);
                                        z.Append(_zdtTemp);
                                        j += 2;
                                    }
                                    else
                                    {
                                        C.__snprintf(_zdtTemp, 4, "%03d", days + 1);
                                        z.Append(_zdtTemp);
                                        j += 3;
                                    }
                                    break;
                                }
                            case 'J':
                                {
                                    C.__snprintf(_zdtTemp, 20, "%.16g", x.JD / 86400000.0);
                                    z.Append(_zdtTemp);
                                    j = z.Length;
                                    break;
                                }
                            case 'm': C.__snprintf(_zdtTemp, 3, "%02d", x.M); z.Append(_zdtTemp); j += 2; break;
                            case 'M': C.__snprintf(_zdtTemp, 3, "%02d", x.m); z.Append(_zdtTemp); j += 2; break;
                            case 's':
                                {
                                    C.__snprintf(_zdtTemp, 30, "%lld", (long)(x.JD / 1000 - 210866760000L));
                                    z.Append(_zdtTemp);
                                    j = z.Length;
                                    break;
                                }
                            case 'S': C.__snprintf(_zdtTemp, 3, "%02d", (int)x.s); z.Append(_zdtTemp); j += 2; break;
                            case 'w':
                                {
                                    z.Append((((x.JD + 129600000) / 86400000) % 7));
                                    break;
                                }
                            case 'Y':
                                {
                                    C.__snprintf(_zdtTemp, 5, "%04d", x.Y);
                                    z.Append(_zdtTemp);
                                    j = z.Length;
                                    break;
                                }
                            default: z.Append('%'); break;
                        }
                    }
                }
                Vdbe.Result_Text(fctx, z, -1, z == (zdtBuf ? DESTRUCTOR_TRANSIENT : DESTRUCTOR_DYNAMIC);
            }
        }

        static void CtimeFunc(FuncContext fctx, int notUsed, Mem[] notUsed2)
        {
            TimeFunc(fctx, 0, null);
        }

        static void CdateFunc(FuncContext fctx, int notUsed, Mem[] notUsed2)
        {
            DateFunc(fctx, 0, null);
        }

        static void CtimestampFunc(FuncContext fctx, int notUsed, Mem[] notUsed2)
        {
            DatetimeFunc(fctx, 0, null);
        }
#else
        static void CurrentTimeFunc(FuncContext fctx, int argc, Mem[] argv)
        {
            string format = (string)sqlite3_user_data(fctx);
            Context ctx = sqlite3_context_db_handle(fctx);
            long iT;
            ctx.Vfs.CurrentTimeInt64(ref iT);
            time_t t = iT / 1000 - 10000 * (long)21086676;
            MutexEx mutex = MutexEx.Alloc(MutexEx.MUTEX.STATIC_MASTER);
            MutexEx.Enter(mutex);
            char zdtBuf;
            tm pTm;
            //pTm = gmtime(&t);
            //strftime(zdtBuf, 20, zFormat, pTm);
            MutexEx.Leave(mutex);

            sqlite3_result_text(fctx, zdtBuf, -1, SQLITE_TRANSIENT);
        }
#endif
        #endregion

        static FuncDef[] _dateTimeFuncs = new FuncDef[]
        {
#if !OMIT_DATETIME_FUNCS
FUNCTION("julianday",        -1, 0, 0, JuliandayFunc ),
FUNCTION("date",             -1, 0, 0, DateFunc      ),
FUNCTION("time",             -1, 0, 0, TimeFunc      ),
FUNCTION("datetime",         -1, 0, 0, DatetimeFunc  ),
FUNCTION("strftime",         -1, 0, 0, StrftimeFunc  ),
FUNCTION("current_time",      0, 0, 0, CtimeFunc     ),
FUNCTION("current_timestamp", 0, 0, 0, CtimestampFunc),
FUNCTION("current_date",      0, 0, 0, CdateFunc     ),
#else
STR_FUNCTION("current_time",      0, "%H:%M:%S",          0, CurrentTimeFunc),
STR_FUNCTION("current_date",      0, "%Y-%m-%d",          0, CurrentTimeFunc),
STR_FUNCTION("current_timestamp", 0, "%Y-%m-%d %H:%M:%S", 0, CurrentTimeFunc),
#endif
};

        static void RegisterDateTimeFunctions()
        {
            FuncDefHash hash = Context.GlobalFunctions;
            FuncDef[] funcs = _dateTimeFuncs;
            for (int i = 0; i < _dateTimeFuncs.Length; i++)
                hash.Insert(_dateTimeFuncs[i]);
        }
    }
}
