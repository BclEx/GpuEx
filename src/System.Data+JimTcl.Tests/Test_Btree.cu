// Code for testing the btree.c module in SQLite.  This code is not included in the SQLite library.  It is used for automated
// testing of the SQLite library.
#include <Core+Btree\Core+Btree.cu.h>
#include <Core+Btree\BtreeInt.cu.h>
#include <Jim.h>

// Usage: sqlite3_shared_cache_report
//
// Return a list of file that are shared and the number of references to each file.
__device__ int sqlite3BtreeSharedCacheReport(ClientData clientData, Jim_Interp *interp, int argc, Jim_Obj *const args[])
{
#ifndef OMIT_SHARED_CACHE
	extern BtShared *sqlite3SharedCacheList;
	Jim_Obj *r = Jim_NewListObj(interp, nullptr, 0);
	for (BtShared *bt = _GLOBAL(BtShared *, sqlite3SharedCacheList); bt; bt = bt->Next)
	{
		const char *file = bt->Pager->get_Filename(true);
		Jim_ListAppendElement(interp, r, Jim_NewStringObj(interp, file, -1));
		Jim_ListAppendElement(interp, r, Jim_NewIntObj(interp, bt->Refs));
	}
	Jim_SetResult(interp, r);
#endif
	return JIM_OK;
}

// Print debugging information about all cursors to standard output.
__device__ void sqlite3BtreeCursorList(Btree *p)
{
#ifdef _DEBUG
	BtShared *bt = p->Bt;
	for (BtCursor *cur = bt->Cursor; cur; cur = cur->Next)
	{
		MemPage *page = cur->Pages[cur->ID];
		char *mode = (cur->WrFlag ? "rw" : "ro");
		_dprintf("CURSOR %p rooted at %4d(%s) currently at %d.%d%s\n", cur, cur->RootID, mode, (page ? page->ID : 0, cur->Idxs[cur->ID]), (cur->State == CURSOR_VALID ? "" : " eof"));
	}
#endif
}
