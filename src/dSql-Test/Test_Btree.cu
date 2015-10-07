
// Code for testing the btree.c module in SQLite.  This code is not included in the SQLite library.  It is used for automated
// testing of the SQLite library.
#include <Core+Btree\BtreeInt.cu.h>
#include <Tcl.h>

// Usage: sqlite3_shared_cache_report
//
// Return a list of file that are shared and the number of references to each file.
__device__ int sqlite3BtreeSharedCacheReport(void *clientData, Tcl_Interp *interp, int argc, char *args[])
{
#ifndef OMIT_SHARED_CACHE
	extern BtShared *sqlite3SharedCacheList;
	BtShared *bt;
	Tcl_Obj *pRet = Tcl_NewObj();
	for(pBt=GLOBAL(BtShared*,sqlite3SharedCacheList); pBt; pBt=pBt->pNext){
		const char *zFile = sqlite3PagerFilename(pBt->pPager, 1);
		Tcl_ListObjAppendElement(interp, pRet, Tcl_NewStringObj(zFile, -1));
		Tcl_ListObjAppendElement(interp, pRet, Tcl_NewIntObj(pBt->nRef));
	}
	Tcl_SetObjResult(interp, pRet);
#endif
	return TCL_OK;
}

// Print debugging information about all cursors to standard output.
__device__ void sqlite3BtreeCursorList(Btree *p)
{
#ifdef _DEBUG
	BtCursor *pCur;
	BtShared *pBt = p->pBt;
	for(pCur=pBt->pCursor; pCur; pCur=pCur->pNext){
		MemPage *pPage = pCur->apPage[pCur->iPage];
		char *zMode = pCur->wrFlag ? "rw" : "ro";
		sqlite3DebugPrintf("CURSOR %p rooted at %4d(%s) currently at %d.%d%s\n",
			pCur, pCur->pgnoRoot, zMode,
			pPage ? pPage->pgno : 0, pCur->aiIdx[pCur->iPage],
			(pCur->eState==CURSOR_VALID) ? "" : " eof"
			);
	}
#endif
}
