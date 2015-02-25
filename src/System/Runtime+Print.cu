#include "Runtime.h"







////////

//	static void renderLogMsg(int iErrCode, const char *zFormat, va_list ap){
//		StrAccum acc;                          /* String accumulator */
//		char zMsg[SQLITE_PRINT_BUF_SIZE*3];    /* Complete log message */
//
//		sqlite3StrAccumInit(&acc, zMsg, sizeof(zMsg), 0);
//		acc.useMalloc = 0;
//		sqlite3VXPrintf(&acc, 0, zFormat, ap);
//		sqlite3GlobalConfig.xLog(sqlite3GlobalConfig.pLogArg, iErrCode,
//			sqlite3StrAccumFinish(&acc));
//	}
//
//	void sqlite3_log(int iErrCode, const char *zFormat, ...){
//		va_list ap;                             /* Vararg list */
//		if( sqlite3GlobalConfig.xLog ){
//			va_start(ap, zFormat);
//			renderLogMsg(iErrCode, zFormat, ap);
//			va_end(ap);
//		}
//	}
//
//#ifdef _DEBUG
//	void sqlite3DebugPrintf(const char *zFormat, ...){
//		va_list ap;
//		StrAccum acc;
//		char zBuf[500];
//		sqlite3StrAccumInit(&acc, zBuf, sizeof(zBuf), 0);
//		acc.useMalloc = 0;
//		va_start(ap,zFormat);
//		sqlite3VXPrintf(&acc, 0, zFormat, ap);
//		va_end(ap);
//		sqlite3StrAccumFinish(&acc);
//		fprintf(stdout,"%s", zBuf);
//		fflush(stdout);
//	}
//#endif
//
//#ifndef OMIT_TRACE
//	void sqlite3XPrintf(StrAccum *p, const char *zFormat, ...){
//		va_list ap;
//		va_start(ap,zFormat);
//		sqlite3VXPrintf(p, 1, zFormat, ap);
//		va_end(ap);
//	}
//#endif
