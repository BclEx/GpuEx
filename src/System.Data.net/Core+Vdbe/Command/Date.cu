// date.c
#include "..\Core+Vdbe.cu.h"
#include <stdlib.h>
#include <time.h>

// On recent Windows platforms, the localtime_s() function is available as part of the "Secure CRT". It is essentially equivalent to 
// localtime_r() available under most POSIX platforms, except that the order of the parameters is reversed.
//
// See http://msdn.microsoft.com/en-us/library/a442x3ye(VS.80).aspx.
//
// If the user has not indicated to use localtime_r() or localtime_s() already, check for an MSVC build environment that provides localtime_s().
#if !defined(HAVE_LOCALTIME_R) && !defined(HAVE_LOCALTIME_S) && defined(_MSC_VER) && defined(_CRT_INSECURE_DEPRECATE)
#define HAVE_LOCALTIME_S 1
#endif

namespace Core { namespace Command
{

#pragma region OMIT_DATETIME_FUNCS
#ifndef OMIT_DATETIME_FUNCS

	struct DateTime
	{
		int64 JD;			// The julian day number times 86400000 */
		int Y, M, D;		// Year, month, and day */
		int h, m;			// Hour and minutes */
		int tz;			// Timezone offset in minutes */
		double s;			// Seconds */
		bool ValidYMD;	// True (1) if Y,M,D are valid
		bool ValidHMS;	// True (1) if h,m,s are valid
		bool ValidJD;		// True (1) if iJD is valid
		bool ValidTZ;		// True (1) if tz is valid
	};

	__device__ static const char *GetDigitsStep(const char *date, int n, int min, int max, char nextC, int *value)
	{
		int val = 0;
		const char *nextDate = nullptr;
		while (n--)
		{
			if (!_isdigit(*date)) goto end_getDigits;
			val = val*10 + *date - '0';
			date++;
		}
		if (val < min || val > max || (nextC != 0 && nextC != *date)) goto end_getDigits;
		nextDate = ++date;

end_getDigits:
		*value = val;
		return nextDate;
	}
	__device__ static int GetDigits(const char *date,
		int n0, int min0, int max0, char nextC0, int *value0)
	{
		date = GetDigitsStep(date, n0, min0, max0, nextC0, value0);
		return (!date ? 0 : 1);
	}
	__device__ static int GetDigits(const char *date,
		int n0, int min0, int max0, char nextC0, int *value0,
		int n1, int min1, int max1, char nextC1, int *value1)
	{
		date = GetDigitsStep(date, n0, min0, max0, nextC0, value0);
		if (!date) return 0;
		date = GetDigitsStep(date, n1, min1, max1, nextC1, value1);
		return (!date ? 1 : 2);
	}
	__device__ static int GetDigits(const char *date,
		int n0, int min0, int max0, char nextC0, int *value0,
		int n1, int min1, int max1, char nextC1, int *value1,
		int n2, int min2, int max2, char nextC2, int *value2)
	{
		date = GetDigitsStep(date, n0, min0, max0, nextC0, value0);
		if (!date) return 0;
		date = GetDigitsStep(date, n1, min1, max1, nextC1, value1);
		if (!date) return 1;
		date = GetDigitsStep(date, n2, min2, max2, nextC2, value2);
		return (!date ? 2 : 3);
	}

	__device__ static bool ParseTimezone(const char *date, DateTime *p)
	{
		while (_isspace(*date)) { date++; }
		p->tz = 0;
		int sgn = 0;
		int c = *date;
		if (c == '-') sgn = -1;
		else if (c == '+') sgn = +1;
		else if (c == 'Z' || c == 'z')
		{
			date++;
			goto zulu_time;
		}
		else return (c != 0);
		date++;
		int hrs, mns;
		if (GetDigits(date, 2, 0, 14, ':', &hrs, 2, 0, 59, 0, &mns) != 2)
			return true;
		date += 5;
		p->tz = sgn*(mns + hrs*60);
zulu_time:
		while (_isspace(*date)) { date++; }
		return (*date != 0);
	}

	__device__ static bool ParseHhMmSs(const char *date, DateTime *p)
	{
		int h, m, s;
		double ms = 0.0;
		if (GetDigits(date, 2, 0, 24, ':', &h, 2, 0, 59, 0, &m) != 2)
			return true;
		date += 5;
		if (*date == ':')
		{
			date++;
			if (GetDigits(date, 2, 0, 59, 0, &s) != 1)
				return true;
			date += 2;
			if (*date == '.' && _isdigit(date[1]))
			{
				double scale = 1.0;
				date++;
				while (_isdigit(*date))
				{
					ms = ms*10.0 + *date - '0';
					scale *= 10.0;
					date++;
				}
				ms /= scale;
			}
		}
		else
			s = 0;
		p->ValidJD = false;
		p->ValidHMS = true;
		p->h = h;
		p->m = m;
		p->s = s + ms;
		if (ParseTimezone(date, p)) return true;
		p->ValidTZ = (p->tz != 0);
		return false;
	}

	__device__ static void ComputeJD(DateTime *p)
	{
		if (p->ValidJD) return;
		int Y, M, D;
		if (p->ValidYMD)
		{
			Y = p->Y;
			M = p->M;
			D = p->D;
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
		int A = Y/100;
		int B = 2 - A + (A/4);
		int X1 = 36525*(Y+4716)/100;
		int X2 = 306001*(M+1)/10000;
		p->JD = (int64)((X1 + X2 + D + B - 1524.5 ) * 86400000);
		p->ValidJD = true;
		if (p->ValidHMS)
		{
			p->JD += p->h*3600000 + p->m*60000 + (int64)(p->s*1000);
			if (p->ValidTZ)
			{
				p->JD -= p->tz*60000;
				p->ValidYMD = false;
				p->ValidHMS = false;
				p->ValidTZ = false;
			}
		}
	}

	__device__ static bool ParseYyyyMmDd(const char *date, DateTime *p)
	{
		bool neg;
		if (date[0] == '-')
		{
			date++;
			neg = true;
		}
		else
			neg = false;
		int Y, M, D;
		if (GetDigits(date,4,0,9999,'-',&Y,2,1,12,'-',&M,2,1,31,0,&D) != 3)
			return true;
		date += 10;
		while (_isspace(*date) || 'T' == *(uint8 *)date) { date++; }
		if (!ParseHhMmSs(date, p)) { } // We got the time
		else if (*date == 0) p->ValidHMS = false;
		else return true;
		p->ValidJD = false;
		p->ValidYMD = true;
		p->Y = (neg ? -Y : Y);
		p->M = M;
		p->D = D;
		if (p->ValidTZ)
			ComputeJD(p);
		return false;
	}

	__device__ static bool SetDateTimeToCurrent(FuncContext *fctx, DateTime *p)
	{
		Context *ctx = Vdbe::Context_Ctx(fctx);
		if (ctx->Vfs->CurrentTimeInt64(&p->JD) == RC_OK)
		{
			p->ValidJD = true;
			return false;
		}
		return true;
	}

	__device__ static bool ParseDateOrTime(FuncContext *fctx, const char *date, DateTime *p)
	{
		double r;
		if (ParseYyyyMmDd(date, p) == 0) return false;
		else if (ParseHhMmSs(date, p) == 0) return false;
		else if (!_strcmp(date, "now")) return SetDateTimeToCurrent(fctx, p);
		else if (ConvertEx::Atof(date, &r, _strlen30(date), TEXTENCODE_UTF8))
		{
			p->JD = (int64)(r*86400000.0 + 0.5);
			p->ValidJD = true;
			return false;
		}
		return true;
	}

	__device__ static void ComputeYMD(DateTime *p)
	{
		if (p->ValidYMD) return;
		if (!p->ValidJD)
		{
			p->Y = 2000;
			p->M = 1;
			p->D = 1;
		}
		else
		{
			int Z = (int)((p->JD + 43200000)/86400000);
			int A = (int)((Z - 1867216.25)/36524.25);
			A = Z + 1 + A - (A/4);
			int B = A + 1524;
			int C = (int)((B - 122.1)/365.25);
			int D = (36525*C)/100;
			int E = (int)((B-D)/30.6001);
			int X1 = (int)(30.6001*E);
			p->D = B - D - X1;
			p->M = (E < 14 ? E-1 : E-13);
			p->Y = (p->M > 2 ? C - 4716 : C - 4715);
		}
		p->ValidYMD = true;
	}

	__device__ static void ComputeHMS(DateTime *p)
	{
		if (p->ValidHMS) return;
		ComputeJD(p);
		int s = (int)((p->JD + 43200000) % 86400000);
		p->s = s/1000.0;
		s = (int)p->s;
		p->s -= s;
		p->h = s/3600;
		s -= p->h*3600;
		p->m = s/60;
		p->s += s - p->m*60;
		p->ValidHMS = true;
	}

	__device__ static void ComputeYMD_HMS(DateTime *p)
	{
		ComputeYMD(p);
		ComputeHMS(p);
	}

	__device__ static void ClearYMD_HMS_TZ(DateTime *p)
	{
		p->ValidYMD = false;
		p->ValidHMS = false;
		p->ValidTZ = false;
	}

#ifndef OMIT_LOCALTIME

	__device__ static int OsLocaltime(time_t *t, tm *tm_)
	{
		int rc;
#if __CUDACC__
		//*tm_ = 0;
		rc = 0;
#elif ((!defined(HAVE_LOCALTIME_R) || !HAVE_LOCALTIME_R) && (!defined(HAVE_LOCALTIME_S) || !HAVE_LOCALTIME_S))
#if THREADSAFE
		MutexEx mutex = MutexEx::Alloc(MutexEx::MUTEX_STATIC_MASTER);
#endif
		MutexEx::Enter(mutex);
		tm *x = localtime(t);
#ifndef OMIT_BUILTIN_TEST
		if (SysEx_GlobalStatics.LocaltimeFault) x = nullptr;
#endif
		if (x) *tm_ = *x;
		MutexEx::Leave(mutex);
		rc = (x == nullptr);
#else
#ifndef OMIT_BUILTIN_TEST
		if (SysEx_GlobalStatics.LocaltimeFault) return 1;
#endif
#if defined(HAVE_LOCALTIME_R) && HAVE_LOCALTIME_R
		rc = (localtime_r(t, tm_) == 0);
#else
		rc = localtime_s(tm_, t);
#endif
#endif
		return rc;
	}

	__device__ static int64 LocaltimeOffset(DateTime *p, FuncContext *fctx, RC *rc)
	{
		// Initialize the contents of sLocal to avoid a compiler warning.
		tm sLocal;
		_memset(&sLocal, 0, sizeof(sLocal));

		DateTime x = *p;
		ComputeYMD_HMS(&x);
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
		ComputeJD(&x);
		time_t t = (time_t)(x.JD/1000 - 21086676*(int64)10000); //210866760000L
		if (OsLocaltime(&t, &sLocal))
		{
			Vdbe::Result_Error(fctx, "local time unavailable", -1);
			*rc = RC_ERROR;
			return 0;
		}
		DateTime y;
		y.Y = sLocal.tm_year + 1900;
		y.M = sLocal.tm_mon + 1;
		y.D = sLocal.tm_mday;
		y.h = sLocal.tm_hour;
		y.m = sLocal.tm_min;
		y.s = sLocal.tm_sec;
		y.ValidYMD = true;
		y.ValidHMS = true;
		y.ValidJD = false;
		y.ValidTZ = false;
		ComputeJD(&y);
		*rc = RC_OK;
		return y.JD - x.JD;
	}
#endif

	__device__ static RC ParseModifier(FuncContext *fctx, const char *mod, DateTime *p)
	{
		RC rc = RC_ERROR;
		int n;
		double r;
		char *z, buf[30];
		z = buf;
		for (n = 0; n < _lengthof(buf)-1 && mod[n]; n++)
			z[n] = (char)_tolower((uint8)mod[n]);
		z[n] = 0;
		switch (z[0])
		{
#ifndef OMIT_LOCALTIME
		case 'l': {
			// localtime - Assuming the current time value is UTC (a.k.a. GMT), shift it to show local time.
			if (!_strcmp(z, "localtime"))
			{
				ComputeJD(p);
				p->JD += LocaltimeOffset(p, fctx, &rc);
				ClearYMD_HMS_TZ(p);
			}
			break; }
#endif
		case 'u': {
			// unixepoch - Treat the current value of p->iJD as the number of seconds since 1970.  Convert to a real julian day number.
			if (!_strcmp(z, "unixepoch") && p->ValidJD)
			{
				p->JD = (p->JD + 43200)/86400 + 21086676*(int64)10000000;
				ClearYMD_HMS_TZ(p);
				rc = RC_OK;
			}
#ifndef OMIT_LOCALTIME
			else if (!_strcmp(z, "utc"))
			{
				ComputeJD(p);
				int64 c1 = LocaltimeOffset(p, fctx, &rc);
				if (rc == RC_OK)
				{
					p->JD -= c1;
					ClearYMD_HMS_TZ(p);
					p->JD += c1 - LocaltimeOffset(p, fctx, &rc);
				}
			}
#endif
			break; }
		case 'w': {
			// weekday N - Move the date to the same time on the next occurrence of weekday N where 0==Sunday, 1==Monday, and so forth.  If the date is already on the appropriate weekday, this is a no-op.
			if (!_strncmp(z, "weekday ", 8) && ConvertEx::Atof(&z[8], &r, _strlen30(&z[8]), TEXTENCODE_UTF8) && (n = (int)r) == r && n >= 0 && r < 7)
			{
				ComputeYMD_HMS(p);
				p->ValidTZ = false;
				p->ValidJD = false;
				ComputeJD(p);
				int64 Z = ((p->JD + 129600000)/86400000) % 7;
				if (Z > n) Z -= 7;
				p->JD += (n - Z)*86400000;
				ClearYMD_HMS_TZ(p);
				rc = RC_OK;
			}
			break; }
		case 's': {
			// start of TTTTT - Move the date backwards to the beginning of the current day, or month or year.
			if (_strncmp(z, "start of ", 9)) break;
			z += 9;
			ComputeYMD(p);
			p->ValidHMS = true;
			p->h = p->m = 0;
			p->s = 0.0;
			p->ValidTZ = false;
			p->ValidJD = false;
			if (!_strcmp(z, "month"))
			{
				p->D = 1;
				rc = RC_OK;
			}
			else if (!_strcmp(z, "year"))
			{
				ComputeYMD(p);
				p->M = 1;
				p->D = 1;
				rc = RC_OK;
			}
			else if (!_strcmp(z, "day"))
				rc = RC_OK;
			break; }
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
		case '9': {
			for (n = 1; z[n] && z[n] != ':' && !_isspace(z[n]); n++) { }
			if (!ConvertEx::Atof(z, &r, n, TEXTENCODE_UTF8))
			{
				rc = RC_ERROR;
				break;
			}
			if (z[n] == ':')
			{
				// A modifier of the form (+|-)HH:MM:SS.FFF adds (or subtracts) the specified number of hours, minutes, seconds, and fractional seconds
				// to the time.  The ".FFF" may be omitted.  The ":SS.FFF" may be omitted.
				const char *z2 = z;
				if (!_isdigit(*z2)) z2++;
				DateTime tx;
				_memset(&tx, 0, sizeof(tx));
				if (ParseHhMmSs(z2, &tx)) break;
				ComputeJD(&tx);
				tx.JD -= 43200000;
				int64 day = tx.JD/86400000;
				tx.JD -= day*86400000;
				if (z[0] == '-') tx.JD = -tx.JD;
				ComputeJD(p);
				ClearYMD_HMS_TZ(p);
				p->JD += tx.JD;
				rc = RC_OK;
				break;
			}
			z += n;
			while (_isspace(*z)) z++;
			n = _strlen30(z);
			if (n > 10 || n < 3) break;
			if (z[n-1] == 's') { z[n-1] = 0; n--; }
			ComputeJD(p);
			rc = RC_OK;
			double rounder = (r < 0 ? -0.5 : +0.5);
			if (n == 3 && !_strcmp(z, "day")) p->JD += (int64)(r*86400000.0 + rounder);
			else if (n == 4 && !_strcmp(z, "hour")) p->JD += (int64)(r*(86400000.0/24.0) + rounder);
			else if (n == 6 && !_strcmp(z, "minute")) p->JD += (int64)(r*(86400000.0/(24.0*60.0)) + rounder);
			else if (n == 6 && !_strcmp(z, "second")) p->JD += (int64)(r*(86400000.0/(24.0*60.0*60.0)) + rounder);
			else if (n == 5 && !_strcmp(z, "month"))
			{
				ComputeYMD_HMS(p);
				p->M += (int)r;
				int x = (p->M > 0 ? (p->M-1)/12 : (p->M-12)/12);
				p->Y += x;
				p->M -= x*12;
				p->ValidJD = false;
				ComputeJD(p);
				int y = (int)r;
				if (y != r)
					p->JD += (int64)((r - y)*30.0*86400000.0 + rounder);
			}
			else if (n == 4 && !_strcmp(z, "year"))
			{
				int y = (int)r;
				ComputeYMD_HMS(p);
				p->Y += y;
				p->ValidJD = false;
				ComputeJD(p);
				if (y != r)
					p->JD += (int64)((r - y)*365.0*86400000.0 + rounder);
			}
			else
				rc = RC_ERROR;
			ClearYMD_HMS_TZ(p);
			break; }
		default: break;
		}
		return rc;
	}

	__device__ static bool IsDate(FuncContext *fctx, int argc, Mem **argv, DateTime *p)
	{
		int i;
		const unsigned char *z;
		_memset(p, 0, sizeof(*p));
		if (argc == 0)
			return SetDateTimeToCurrent(fctx, p);
		TYPE type;
		if ((type = Vdbe::Value_Type(argv[0])) == TYPE_FLOAT || type == TYPE_INTEGER)
		{
			p->JD = (int64)(Vdbe::Value_Double(argv[0])*86400000.0 + 0.5);
			p->ValidJD = true;
		}
		else
		{
			z = Vdbe::Value_Text(argv[0]);
			if (!z || ParseDateOrTime(fctx, (char *)z, p)) return true;
		}
		for (i = 1; i < argc; i++)
		{
			z = Vdbe::Value_Text(argv[i]);
			if (!z || ParseModifier(fctx, (char *)z, p)) return true;
		}
		return false;
	}

	__device__ static void JuliandayFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		DateTime x;
		if (!IsDate(fctx, argc, argv, &x))
		{
			ComputeJD(&x);
			Vdbe::Result_Double(fctx, x.JD/86400000.0);
		}
	}

	__device__ static void DatetimeFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		DateTime x;
		if (!IsDate(fctx, argc, argv, &x))
		{
			char buf[100];
			ComputeYMD_HMS(&x);
			__snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d", x.Y, x.M, x.D, x.h, x.m, (int)(x.s));
			Vdbe::Result_Text(fctx, buf, -1, DESTRUCTOR_TRANSIENT);
		}
	}

	__device__ static void TimeFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		DateTime x;
		if (!IsDate(fctx, argc, argv, &x))
		{
			char buf[100];
			ComputeHMS(&x);
			__snprintf(buf, sizeof(buf), "%02d:%02d:%02d", x.h, x.m, (int)x.s);
			Vdbe::Result_Text(fctx, buf, -1, DESTRUCTOR_TRANSIENT);
		}
	}

	__device__ static void DateFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		DateTime x;
		if (!IsDate(fctx, argc, argv, &x))
		{
			char buf[100];
			ComputeYMD(&x);
			__snprintf(buf, sizeof(buf), "%04d-%02d-%02d", x.Y, x.M, x.D);
			Vdbe::Result_Text(fctx, buf, -1, DESTRUCTOR_TRANSIENT);
		}
	}

	__device__ static void StrftimeFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		DateTime x;
		uint64 n;
		size_t i, j;
		char *z;
		const char *fmt = (const char *)Vdbe::Value_Text(argv[0]);
		char buf[100];
		if (!fmt || IsDate(fctx, argc-1, argv+1, &x)) return;
		Context *ctx = Vdbe::Context_Ctx(fctx);
		for (i = 0, n = 1; fmt[i]; i++, n++)
		{
			if (fmt[i] == '%')
			{
				switch (fmt[i+1])
				{
				case 'd':
				case 'H':
				case 'm':
				case 'M':
				case 'S':
				case 'W':
					n++;
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
		ASSERTCOVERAGE(n == sizeof(buf)-1);
		ASSERTCOVERAGE(n == sizeof(buf));
		ASSERTCOVERAGE(n == (uint64)ctx->Limits[LIMIT_LENGTH]+1);
		ASSERTCOVERAGE(n == (uint64)ctx->Limits[LIMIT_LENGTH]);
		if (n < sizeof(buf))
			z = buf;
		else if (n > (uint64)ctx->Limits[LIMIT_LENGTH])
		{
			Vdbe::Result_ErrorOverflow(fctx);
			return;
		}
		else
		{
			z = (char *)_tagalloc(ctx, (int)n);
			if (!z)
			{
				Vdbe::Result_ErrorNoMem(fctx);
				return;
			}
		}
		ComputeJD(&x);
		ComputeYMD_HMS(&x);
		for (i = j = 0; fmt[i]; i++)
		{
			if (fmt[i] != '%')
				z[j++] = fmt[i];
			else
			{
				i++;
				switch (fmt[i])
				{
				case 'd': __snprintf(&z[j], 3, "%02d", x.D); j+=2; break;
				case 'f': {
					double s = x.s;
					if (s > 59.999) s = 59.999;
					__snprintf(&z[j], 7, "%06.3f", s);
					j += _strlen30(&z[j]);
					break; }
				case 'H': __snprintf(&z[j], 3, "%02d", x.h); j+=2; break;
				case 'W': // Fall thru
				case 'j': {
					DateTime y = x;
					y.ValidJD = false;
					y.M = 1;
					y.D = 1;
					ComputeJD(&y);
					int days = (int)((x.JD-y.JD+43200000)/86400000); // Number of days since 1st day of year
					if (fmt[i] == 'W')
					{
						int wd = (int)(((x.JD+43200000)/86400000)%7); // 0=Monday, 1=Tuesday, ... 6=Sunday
						__snprintf(&z[j], 3, "%02d", (days+7-wd)/7);
						j += 2;
					}
					else
					{
						__snprintf(&z[j], 4, "%03d", days+1);
						j += 3;
					}
					break; }
				case 'J': {
					__snprintf(&z[j], 20, "%.16g", x.JD/86400000.0);
					j += _strlen30(&z[j]);
					break; }
				case 'm': __snprintf(&z[j], 3, "%02d", x.M); j+=2; break;
				case 'M': __snprintf(&z[j], 3, "%02d", x.m); j+=2; break;
				case 's': {
					__snprintf(&z[j], 30, "%lld", (int64)(x.JD/1000 - 21086676*(int64)10000));
					j += _strlen30(&z[j]);
					break; }
				case 'S':  __snprintf(&z[j], 3, "%02d",(int)x.s); j+=2; break;
				case 'w': {
					z[j++] = (char)(((x.JD+129600000)/86400000) % 7) + '0';
					break; }
				case 'Y': {
					__snprintf(&z[j], 5, "%04d", x.Y); j += _strlen30(&z[j]);
					break; }
				default: z[j++] = '%'; break; }
			}
		}
		z[j] = 0;
		Vdbe::Result_Text(fctx, z, -1, (z == buf ? DESTRUCTOR_TRANSIENT : DESTRUCTOR_DYNAMIC));
	}

	__device__ static void CtimeFunc(FuncContext *fctx, int notUsed, Mem **notUsed2)
	{
		TimeFunc(fctx, 0, 0);
	}

	__device__ static void CdateFunc(FuncContext *fctx, int notUsed, Mem **notUsed2)
	{
		DateFunc(fctx, 0, 0);
	}

	__device__ static void CtimestampFunc(FuncContext *fctx, int notUsed, Mem **notUsed2)
	{
		DatetimeFunc(fctx, 0, 0);
	}

#else

	__device__ static void CurrentTimeFunc(FuncContext *fctx, int argc, Mem **argv)
	{
		char *format = (char *)Vdbe::User_Data(fctx);
		Context *ctx = Vdbe::Context_Ctx(fctx);
		int64 now;
		if (ctx->Vfs->CurrentTimeInt64(&now)) return;
		time_t t = now/1000 - 10000*(int64)21086676;
		tm *tm_;
		tm sNow;
#ifdef HAVE_GMTIME_R
		tm_ = gmtime_r(&t, &sNow);
#else
		MutexEx mutex = MutexEx::Alloc(MUTEX_STATIC_MASTER);
		MutexEx::Enter(mutex);
		tm_ = gmtime(&t);
		if (tm_) _memcpy(&sNow, tm_, sizeof(sNow));
		MutexEx::Leave(mutex);
#endif
		if (tm_)
		{
			char buf[20];
			strftime(buf, 20, format, &sNow);
			Vbde::Result_Text(fctx, buf, -1, DESTRUCTUR_TRANSIENT);
		}
	}

#endif
#pragma endregion

	__device__ static _WSD FuncDef g_dateTimeFuncs[] =
	{
#ifndef OMIT_DATETIME_FUNCS
		FUNCTION(julianday,        -1, 0, 0, JuliandayFunc ),
		FUNCTION(date,             -1, 0, 0, DateFunc      ),
		FUNCTION(time,             -1, 0, 0, TimeFunc      ),
		FUNCTION(datetime,         -1, 0, 0, DatetimeFunc  ),
		FUNCTION(strftime,         -1, 0, 0, StrftimeFunc  ),
		FUNCTION(current_time,      0, 0, 0, CtimeFunc     ),
		FUNCTION(current_timestamp, 0, 0, 0, CtimestampFunc),
		FUNCTION(current_date,      0, 0, 0, CdateFunc     ),
#else
		STR_FUNCTION(current_time,      0, "%H:%M:%S",          0, CurrentTimeFunc),
		STR_FUNCTION(current_date,      0, "%Y-%m-%d",          0, CurrentTimeFunc),
		STR_FUNCTION(current_timestamp, 0, "%Y-%m-%d %H:%M:%S", 0, CurrentTimeFunc),
#endif
	};

	__device__ void Date_::RegisterDateTimeFunctions()
	{
		FuncDefHash *hash = &Main_GlobalFunctions;
		FuncDef *funcs = (FuncDef *)&_GLOBAL(FuncDef, g_dateTimeFuncs);
		for (int i = 0; i < _lengthof(g_dateTimeFuncs); i++)
			Callback::FuncDefInsert(hash, &funcs[i]);
	}

} }
